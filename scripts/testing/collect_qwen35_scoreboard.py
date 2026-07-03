"""Collect the qwen3.5 scoreboard rows from run directories.

Emits the fix-doc reporting format:
  fwd_s bwd_s opt_s step_s fwd_H bwd_H step_H RAM loss grad_norm

Usage: .venv/bin/python scripts/testing/collect_qwen35_scoreboard.py LABEL=RUN_DIR [...]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def parse_lat(lat: Path | None) -> dict[str, float]:
    out = {"fwd_s": float("nan"), "bwd_s": float("nan"), "opt_s": float("nan"), "step_s": float("nan")}
    if not lat or not lat.exists():
        return out
    text = lat.read_text()
    def grab(pat):
        m = re.search(pat + r"\s*\|\s*([0-9.]+)", text)
        return float(m.group(1)) / 1000.0 if m else float("nan")
    out["fwd_s"] = grab(r"step\.forward")
    out["bwd_s"] = grab(r"step\.backward")
    out["opt_s"] = grab(r"optimizer/update side = e2e measured - fwd/bwd")
    out["step_s"] = grab(r"trainer e2e measured step incl optimizer")
    return out


def parse_phases(jsonl: Path | None) -> dict[str, float]:
    out = {"fwd_H": float("nan"), "bwd_H": float("nan")}
    if not jsonl or not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        r = json.loads(line)
        v = r.get("peak_allocated_within_phase", 0)
        v = (v.get("bytes", 0) if isinstance(v, dict) else v) or 0
        if r.get("phase") == "after_forward":
            out["fwd_H"] = v / 2**30
        elif r.get("phase") == "after_backward":
            out["bwd_H"] = v / 2**30
    return out


def parse_mem(md: Path | None) -> float:
    if not md or not md.exists():
        return float("nan")
    m = re.search(r"measured_training_step_peak_allocated_hbm_bytes \| ([0-9.]+)", md.read_text())
    return float(m.group(1)) / 1024.0 if m else float("nan")


def parse_loss(log: Path | None) -> tuple[str, str]:
    if not log or not log.exists():
        return "-", "-"
    txt = subprocess.run(["grep", "-oh", r"'train_loss': '[^']*'\|'grad_norm': '[^']*'", str(log)],
                         capture_output=True, text=True).stdout.strip().splitlines()
    loss = gn = "-"
    for line in txt:
        if "train_loss" in line:
            loss = line.split("'")[3]
        if "grad_norm" in line:
            gn = line.split("'")[3]
    return loss, gn


def parse_ram(root: Path) -> float:
    pm = find(root, "process_memory.csv")
    if not pm:
        return float("nan")
    best = 0.0
    for line in pm.read_text().splitlines()[1:]:
        parts = line.split(",")
        try:
            best = max(best, float(parts[4]))
        except (ValueError, IndexError):
            continue
    return best / 2**30


def main() -> None:
    print(f"{'row':44s} {'fwd_s':>7} {'bwd_s':>7} {'opt_s':>6} {'step_s':>7} {'fwd_H':>7} {'bwd_H':>7} {'step_H':>8} {'RAM':>6} {'loss':>7} {'grad_norm':>9}")
    for spec in sys.argv[1:]:
        label, _, path = spec.partition("=")
        root = Path(path)
        lat = parse_lat(find(root, "lat.md"))
        ph = parse_phases(find(root, "memory_breakdown.jsonl"))
        step_h = parse_mem(find(root, "memory.md"))
        loss, gn = parse_loss(find(root, "train.log"))
        ram = parse_ram(root)
        print(f"{label:44s} {lat['fwd_s']:7.1f} {lat['bwd_s']:7.1f} {lat['opt_s']:6.1f} {lat['step_s']:7.1f} "
              f"{ph['fwd_H']:7.1f} {ph['bwd_H']:7.1f} {step_h/1.0:8.1f} {ram:6.0f} {loss:>7} {gn:>9}")


if __name__ == "__main__":
    main()
