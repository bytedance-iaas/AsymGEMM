#!/usr/bin/env python3
"""Show LF profiling run status as one table per model.

Scans a profiling output root (default: profiling_both) and prints, for every
model found, a table of   Model | Workload | Backend | Config | Status .

Status is derived from each run's artifacts (profile.json / train.log) and, for
a job that just started and hasn't written a log yet, from the live process list.

Usage:
    show_status.py [PROFILING_DIR]        # default profiling_both
    show_status.py profiling
    show_status.py /abs/path/to/profiling_both
"""
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"combined", "metrics", "memory_combined", "c2c_combined", "utilization_combined", "throughput_combined"}
OOM_GPU_MARKERS = ("CUDA out of memory", "OutOfMemoryError")
OOM_HOST_MARKERS = (
    "HOST_OOM_EVIDENCE=true",
    "Out of memory: Killed process",
    "Memory cgroup out of memory",
    "cgroup out of memory",
)

# Higher wins when the same logical config has several dirs (nsys/source, stale roots).
PRIORITY = {
    "OK": 6, "RUNNING": 6, "OOM (GPU)": 4, "OOM (host RAM)": 4,
    "FAILED (non-OOM)": 3, "INCOMPLETE": 2, "NOT RUN": 1,
}
CONFIG_ORDER = [
    "none (no offload)", "none+expert-offload", "none+exp+attn-offload",
    "none+exp+attn-offload+layerGC", "GC-experts", "GC-attn+experts", "GC-layer",
]


def model_tag(name_or_path: str) -> str:
    return name_or_path.split("/")[-1].lower()


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
    for t in tokens:
        if t.startswith(prefix):
            return t[len(prefix):]
    return default


def priority(status: str) -> int:
    if status.startswith("SIG"):
        return 3
    return PRIORITY.get(status, 0)


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
        with open(log, "r", errors="ignore") as fh:
            if size > 2_000_000:
                fh.seek(size - 2_000_000)
            text = fh.read()
    except OSError:
        text = ""
    if any(m in text for m in OOM_GPU_MARKERS):
        return "OOM (GPU)", mtime
    if any(m in text for m in OOM_HOST_MARKERS):
        return "OOM (host RAM)", mtime
    for code in reversed(_failure_codes(text)):
        signal_status = _signal_status(code)
        if signal_status is not None:
            return signal_status, mtime
    if any(m in text for m in ("Training command failed", "Traceback", "ChildFailedError", "failed with status")):
        return "FAILED (non-OOM)", mtime
    if time.time() - mtime < 600:
        return "RUNNING", mtime
    return "INCOMPLETE", mtime


def collect(root: Path) -> dict:
    # First pass: one entry per (logical config, config_root). Within a single run the
    # nsys/source pair is merged (OK if either, else best status). Second pass: collapse
    # config_roots that differ only in warmup/steps/rank/etc., defaulting to the one with the
    # most max_steps (ties: newest run) so a short smoke run can't hide the real measurement.
    runs: dict[tuple, dict] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cr in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
            model = cr.name.split("__gpus")[0]
            m = re.search(r"b(\d+)_s(\d+)_ga(\d+)", cr.name)
            if not model or not m:
                continue
            batch, seq, ga = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ws = re.search(r"_w\d+_s(\d+)", cr.name)  # max_steps from the config-root tag
            steps = int(ws.group(1)) if ws else 0
            for rd in sorted(p for p in cr.iterdir() if p.is_dir()):
                toks = rd.name.split("__")
                if len(toks) < 3 or toks[1] not in ("nsys", "source"):
                    continue
                backend, recompute = toks[0], toks[2]
                config = config_label(
                    _tok(toks, "pol"),
                    _tok(toks, "expact", "0") == "1",
                    _tok(toks, "attnact", "0") == "1",
                    _tok(toks, "layeract", "0") == "1",
                    _tok(toks, "layergc", "0") == "1",
                    _tok(toks, "sdparecomp", "0") == "1",
                )
                leaf = next((p for p in rd.iterdir() if p.is_dir() and p.name.startswith("b")), None)
                status, mtime = classify(leaf)
                lkey = (model, seq, batch, ga, f"{backend} ({recompute})", config)
                rank = (status == "OK", priority(status))
                run = runs.get((lkey, cr.name))
                if run is None:
                    runs[(lkey, cr.name)] = {"lkey": lkey, "rank": rank, "status": status, "mtime": mtime, "steps": steps}
                else:
                    if rank > run["rank"]:
                        run["rank"], run["status"] = rank, status
                    run["mtime"] = max(run["mtime"], mtime)
    rows: dict[tuple, list] = {}
    pick: dict[tuple, tuple] = {}  # logical key -> (max_steps, mtime) of the chosen run
    for run in runs.values():
        lk = run["lkey"]
        cur = (run["steps"], run["mtime"])
        if lk not in pick or cur > pick[lk]:
            pick[lk] = cur
            rows[lk] = [priority(run["status"]), run["mtime"], run["status"]]
    return rows


def overlay_running(rows: dict, root: Path) -> None:
    """Mark the live job as RUNNING even if it has not written a train.log yet.

    Only applies when the live job's output path is under `root`, so querying one
    profiling dir never shows a job that is actually writing to a different one.
    """
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return
    root_str = str(root.resolve())
    for line in out.splitlines():
        if "run_lf_profiled_train.py" not in line or " grep " in line:
            continue

        def arg(name: str, d: str = "") -> str:
            mo = re.search(rf"--{name}\s+(\S+)", line)
            return mo.group(1) if mo else d

        def env(name: str, d: str = "") -> str:
            mo = re.search(rf"(?:^|\s){name}=(\S+)", line)
            return mo.group(1) if mo else d

        out_dir = arg("output_dir") or env("OUT_DIR") or env("PROFILE_OUTPUT_DIR")
        if not out_dir or root_str not in out_dir:
            continue  # this job writes to a different profiling root (or path unknown)
        mpath = arg("model_name_or_path") or env("MODEL_NAME_OR_PATH")
        seq = arg("cutoff_len") or env("CUTOFF_LEN")
        batch = arg("per_device_train_batch_size") or env("PER_DEVICE_TRAIN_BATCH_SIZE")
        if not (mpath and seq and batch):
            continue
        ga = int(arg("gradient_accumulation_steps") or env("GRADIENT_ACCUMULATION_STEPS") or "1")
        backend = env("BACKEND") or ("zero3_offload" if "ds_z3" in line else "asym_cpuadamwds")
        gc = (arg("gradient_checkpointing") or env("GRADIENT_CHECKPOINTING") or "false").lower()
        recompute = "recomp" if gc == "true" else "norecomp"
        config = config_label(
            arg("asym_expert_recompute_policy") or env("ASYM_EXPERT_RECOMPUTE_POLICY") or "none",
            env("ASYMM_EXPERT_ACT_OFFLOAD") == "true",
            env("ASYMM_ATTN_ACT_OFFLOAD") == "true",
            env("ASYMM_LAYER_ACT_OFFLOAD") == "true",
            env("ASYMM_LAYER_GC") == "true",
        )
        key = (model_tag(mpath), int(seq), int(batch), ga, f"{backend} ({recompute})", config)
        rows[key] = [PRIORITY["RUNNING"], time.time(), "RUNNING"]


def print_tables(root: Path, rows: dict) -> None:
    cfg_rank = {c: i for i, c in enumerate(CONFIG_ORDER)}
    by_model: dict[str, list] = {}
    for (model, seq, batch, ga, be, config), (_prio, _mt, status) in rows.items():
        wl = f"s{seq}·b{batch}" + (f"·ga{ga}" if ga != 1 else "")
        by_model.setdefault(model, []).append((seq, batch, be, config, [model, wl, be, config, status]))
    print(f"Profiling status: {root}\n")
    for model in sorted(by_model):
        recs = sorted(by_model[model], key=lambda r: (r[0], r[1], r[2], cfg_rank.get(r[3], 99), r[3]))
        data = [r[4] for r in recs]
        head = ["Model", "Workload", "Backend", "Config", "Status"]
        w = [max(len(head[i]), max((len(d[i]) for d in data), default=0)) for i in range(5)]
        print("  ".join(head[i].ljust(w[i]) for i in range(5)))
        print("  ".join("-" * w[i] for i in range(5)))
        for d in data:
            print("  ".join(d[i].ljust(w[i]) for i in range(5)))
        cnt = Counter(d[4] for d in data)
        print(f"  ({len(data)} configs — " + ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items())) + ")\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Show LF profiling run status, one table per model.")
    ap.add_argument("root", nargs="?", default="profiling_both",
                    help="profiling output dir (default: profiling_both); relative to repo root or absolute")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / args.root
    if not root.is_dir():
        sys.exit(f"show_status: not a directory: {root}")
    rows = collect(root)
    overlay_running(rows, root)
    if not rows:
        print(f"No runs found under {root}")
        return
    print_tables(root, rows)


if __name__ == "__main__":
    main()
