#!/usr/bin/env python3
"""Aggregate per-rank artifacts of a multi-rank (DP) row into one dp_row.json.

Two modes (gb200_dp.md D0/D1/D2):
  --run-dir DIR          torchrun row: DIR holds rank0's profile.json (source_profile.json)
                         plus rank<R>_memstats.json written by every rank.
  --rank-dir DIR ...     independent-pair probe (run_dp2_pair.sh): one artifact dir per rank,
                         each with its own profile.json.

Emits {out}: per-rank {step_s fwd_s bwd_s opt_s step_H rss loss watchdog_fired}, wall step_s
(= max rank), summed RSS, and audit fields. Timing follows show_metrics.py: per-step averages
over measured step_samples.rows (is_warmup==False; first/last measured dropped when >=3 remain);
falls back to trainer.timing.measured_e2e when samples are absent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

GIB = 1024**3


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_profile(run_dir: Path) -> Path | None:
    for name in ("source_profile.json", "profile.json"):
        hits = sorted(run_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def _step_averages(profile: dict[str, Any]) -> dict[str, float | None]:
    rows = (((profile.get("step_samples") or {}).get("rows")) or [])
    measured = [r for r in rows if isinstance(r, dict) and not r.get("is_warmup")]
    if len(measured) >= 3:
        measured = measured[1:-1]
    out: dict[str, float | None] = {"fwd_s": None, "bwd_s": None, "opt_s": None, "step_s": None}
    if measured:
        def avg(key: str) -> float | None:
            vals = [r.get(key) for r in measured if isinstance(r.get(key), (int, float))]
            return (sum(vals) / len(vals) / 1000.0) if vals else None

        out["fwd_s"] = avg("forward_milliseconds")
        out["bwd_s"] = avg("backward_milliseconds")
        out["opt_s"] = avg("optimizer_milliseconds")
        if all(out[k] is not None for k in ("fwd_s", "bwd_s", "opt_s")):
            out["step_s"] = out["fwd_s"] + out["bwd_s"] + out["opt_s"]  # type: ignore[operator]
    if out["step_s"] is None:
        timing = ((profile.get("trainer") or {}).get("timing")) or {}
        ms = timing.get("measured_e2e_step_milliseconds")
        steps = timing.get("measured_steps")
        if isinstance(ms, (int, float)) and isinstance(steps, (int, float)) and steps:
            out["step_s"] = float(ms) / float(steps) / 1000.0
    return out


def _rank_record_from_profile(profile: dict[str, Any], tag: str) -> dict[str, Any]:
    memory = profile.get("memory") or {}
    gpu = memory.get("gpu") or {}
    process = memory.get("process") or {}
    losses = ((profile.get("trainer") or {}).get("losses")) or []
    rec: dict[str, Any] = {"source": tag}
    rec.update(_step_averages(profile))
    rec["step_H_bytes"] = gpu.get("peak_allocated_hbm_bytes") or memory.get("peak_allocated_hbm_bytes")
    rec["rss_peak_bytes"] = process.get("rss_peak_bytes")
    rec["losses"] = [
        {"step": l.get("measured_step"), "loss": l.get("loss"), "is_warmup": l.get("is_warmup")}
        for l in losses
        if isinstance(l, dict)
    ]
    config = profile.get("config") or {}
    rec["backend"] = config.get("backend")
    rec["global_batch_size"] = config.get("global_batch_size")
    rec["per_device_train_batch_size"] = config.get("per_device_train_batch_size") or config.get("batch_size")
    return rec


def _watchdog_sentinels(root: Path) -> list[str]:
    return [str(p) for p in root.rglob("*.host_mem_watchdog_fired")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="torchrun artifact dir (one profile.json + rank<R>_memstats.json)")
    parser.add_argument("--rank-dir", type=Path, action="append", default=[], help="independent per-rank artifact dir (repeatable)")
    parser.add_argument("--external-csv", type=Path, default=None, help="optional external sampler csv (ts,pid,rss_kb,gpu_uuid_mem)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    row: dict[str, Any] = {"label": args.label, "ranks": [], "watchdog_sentinels": [], "notes": []}

    if args.run_dir is not None:
        profile_path = _find_profile(args.run_dir)
        if profile_path is None:
            row["notes"].append(f"no profile.json under {args.run_dir}")
        else:
            profile = _load_json(profile_path)
            if profile:
                rec = _rank_record_from_profile(profile, str(profile_path))
                rec["rank"] = 0
                row["ranks"].append(rec)
        for ms_path in sorted(args.run_dir.rglob("rank*_memstats.json")):
            ms = _load_json(ms_path)
            if not ms:
                continue
            rank = ms.get("rank")
            existing = next((r for r in row["ranks"] if r.get("rank") == rank), None)
            devices = ms.get("devices") or []
            peak = max((d.get("peak_allocated_hbm_bytes") or 0 for d in devices), default=None)
            rss = (ms.get("process_memory") or {}).get("rss_peak_bytes")
            if existing is None:
                row["ranks"].append(
                    {
                        "rank": rank,
                        "source": str(ms_path),
                        "step_H_bytes": peak,
                        "rss_peak_bytes": rss,
                        "current_device": ms.get("current_device"),
                        "cuda_visible_devices": ms.get("cuda_visible_devices"),
                    }
                )
            else:
                existing["memstats"] = {
                    "step_H_bytes": peak,
                    "rss_peak_bytes": rss,
                    "current_device": ms.get("current_device"),
                    "cuda_visible_devices": ms.get("cuda_visible_devices"),
                }
        row["watchdog_sentinels"] += _watchdog_sentinels(args.run_dir)

    for index, rank_dir in enumerate(args.rank_dir):
        profile_path = _find_profile(rank_dir)
        if profile_path is None:
            row["ranks"].append({"rank": index, "source": str(rank_dir), "error": "no profile.json"})
            continue
        profile = _load_json(profile_path)
        if not profile:
            row["ranks"].append({"rank": index, "source": str(profile_path), "error": "unreadable profile.json"})
            continue
        rec = _rank_record_from_profile(profile, str(profile_path))
        rec["rank"] = index
        row["ranks"].append(rec)
        row["watchdog_sentinels"] += _watchdog_sentinels(rank_dir)

    if args.external_csv is not None and args.external_csv.is_file():
        per_pid_rss: dict[str, int] = {}
        gpu_mem: dict[str, int] = {}
        for line in args.external_csv.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            _, pid, rss_kb, gpu = parts[0], parts[1], parts[2], parts[3]
            try:
                rss = int(rss_kb) * 1024
            except ValueError:
                rss = 0
            if rss:
                per_pid_rss[pid] = max(per_pid_rss.get(pid, 0), rss)
            if gpu and ":" in gpu:
                uuid, mem = gpu.rsplit(":", 1)
                mem_mib = mem.strip().split(" ")[0]
                try:
                    key = f"{pid}@{uuid}"
                    gpu_mem[key] = max(gpu_mem.get(key, 0), int(mem_mib) * 1024 * 1024)
                except ValueError:
                    pass
        row["external_sampler"] = {
            "per_pid_rss_peak_bytes": per_pid_rss,
            "per_pid_gpu_mem_peak_bytes": gpu_mem,
        }

    step_values = [r.get("step_s") for r in row["ranks"] if isinstance(r.get("step_s"), (int, float))]
    rss_values = [r.get("rss_peak_bytes") for r in row["ranks"] if isinstance(r.get("rss_peak_bytes"), (int, float))]
    row["wall_step_s"] = max(step_values) if step_values else None
    row["summed_rss_peak_bytes"] = sum(rss_values) if rss_values else None
    row["summed_rss_peak_gib"] = round(row["summed_rss_peak_bytes"] / GIB, 1) if row["summed_rss_peak_bytes"] else None
    row["rank_count"] = len(row["ranks"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: row[k] for k in ("label", "rank_count", "wall_step_s", "summed_rss_peak_gib")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
