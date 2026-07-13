#!/usr/bin/env python3
"""Summarize DeepSpeed NVMe-offload cost from an nsys sqlite (+ optional md0 io samples).

Produces a small, flat, agent-parseable ``offload_io.json`` next to ``profile.json`` and, when
``profile.json`` is small enough, folds the same block in under the ``offload_io`` key.

Why this exists: the nsys trace is captured with ``--trace=cuda,nvtx,osrt`` and capture-range
``cudaProfilerApi``, so it already covers only the measured steps. But the raw sqlite is hundreds
of MB. This collapses it to the numbers an agent actually wants:

  gpu_kernel_ms          sum of CUDA kernel durations (compute)
  gpu_busy_union_ms      merged union of kernel+memcpy intervals (true GPU-busy wall)
  gpu_span_ms            max(end)-min(start) over kernel+memcpy (measured GPU wall)
  gpu_idle_ms            gpu_span_ms - gpu_busy_union_ms  (stall; ~param swap-in wait)
  memcpy.<dir>_ms/_gb    CPU<->GPU copies by direction (h2d=param loads, d2h=evict/offload)
  nvtx.offload_wait_ms   PartitionedParameterCoordinator.fetch_sub_module total
                         (time blocked on offloaded params; dominated by NVMe swap-in)
  nvtx.all_gather_ms     __all_gather_params (CPU/pinned -> GPU gather)
  nvtx.prefetch_nvme_ms  __prefetch_nvme_param_partitions (async NVMe swap-in issue)
  nvtx.forward_ms/backward_ms/step_total_ms   step totals for context
  nvme.read_gb/write_gb, nvme.read_gbps_peak/_median_active   from /sys/block/<dev>/stat samples

Usage:
  summarize_nvme_offload.py TRACE.sqlite [--io-samples io_samples.csv]
      [--profile-json profile.json] [--output-json offload_io.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from statistics import median


# nsys/CUPTI memcpy copyKind codes -> direction label (h2d/d2h are the stable, important ones).
_COPYKIND = {1: "h2d", 2: "d2h", 3: "h2h", 4: "d2d", 8: "d2d", 10: "p2p", 11: "p2p"}


def _table_exists(cur, name: str) -> bool:
    return cur.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone() is not None


def _merge_union_ms(intervals) -> tuple[float, float]:
    """Return (union_ms, span_ms) for a list of (start_ns, end_ns)."""
    iv = sorted(i for i in intervals if i[0] is not None and i[1] is not None and i[1] > i[0])
    if not iv:
        return 0.0, 0.0
    span = iv[-1][1] - iv[0][0]
    union = 0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            union += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    union += ce - cs
    return union / 1e6, span / 1e6


def nvtx_totals(cur) -> dict:
    """Sum NVTX range wall-time (ms) per resolved name, then pick the ranges we care about."""
    if not _table_exists(cur, "NVTX_EVENTS"):
        return {}
    rows = cur.execute(
        """select coalesce(text,(select value from StringIds where id=textId)) as nm,
                  sum(end-start)/1e6 as ms
           from NVTX_EVENTS where end is not null group by nm"""
    ).fetchall()
    by = {nm: ms for nm, ms in rows if nm}

    def pick(needle: str) -> float:
        return round(sum(ms for nm, ms in by.items() if needle in nm), 3)

    return {
        "offload_wait_ms": pick("fetch_sub_module"),
        "all_gather_ms": pick("__all_gather_params"),
        "prefetch_nvme_ms": pick("__prefetch_nvme_param_partitions"),
        "forward_ms": pick("step.forward"),
        "backward_ms": pick("step.backward"),
        "step_total_ms": pick("lf.step.total"),
    }


def gpu_totals(cur) -> dict:
    out: dict = {}
    kern = []
    if _table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        kern = cur.execute(
            "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where end is not null"
        ).fetchall()
        out["gpu_kernel_ms"] = round(sum(e - s for s, e in kern) / 1e6, 3)
        out["gpu_kernel_count"] = len(kern)

    memcpy = []
    if _table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        rows = cur.execute(
            "select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end is not null"
        ).fetchall()
        agg: dict = {}
        for s, e, kind, nbytes in rows:
            memcpy.append((s, e))
            d = _COPYKIND.get(kind, f"kind_{kind}")
            a = agg.setdefault(d, [0.0, 0])
            a[0] += (e - s) / 1e6
            a[1] += nbytes or 0
        mc: dict = {}
        for d, (ms, nbytes) in sorted(agg.items()):
            mc[f"{d}_ms"] = round(ms, 3)
            mc[f"{d}_gb"] = round(nbytes / 1e9, 4)
        out["memcpy"] = mc

    union_ms, span_ms = _merge_union_ms(kern + memcpy)
    out["gpu_busy_union_ms"] = round(union_ms, 3)
    out["gpu_span_ms"] = round(span_ms, 3)
    out["gpu_idle_ms"] = round(max(0.0, span_ms - union_ms), 3)
    return out


def io_totals(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    ts, sr, sw = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["epoch_s"]))
                sr.append(int(row["sectors_read"]))
                sw.append(int(row["sectors_written"]))
            except (KeyError, ValueError):
                continue
    if len(ts) < 2:
        return {}
    SECTOR = 512
    read_gb = (sr[-1] - sr[0]) * SECTOR / 1e9
    write_gb = (sw[-1] - sw[0]) * SECTOR / 1e9
    rrates, wrates = [], []
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            continue
        rrates.append((sr[i] - sr[i - 1]) * SECTOR / 1e9 / dt)
        wrates.append((sw[i] - sw[i - 1]) * SECTOR / 1e9 / dt)
    active = [r for r in rrates if r > 0.2]  # GB/s; drop idle gaps
    return {
        "device_samples": len(ts),
        "duration_s": round(ts[-1] - ts[0], 1),
        "read_gb": round(read_gb, 3),
        "write_gb": round(write_gb, 3),
        "read_gbps_peak": round(max(rrates), 3) if rrates else 0.0,
        "read_gbps_median_active": round(median(active), 3) if active else 0.0,
        "write_gbps_peak": round(max(wrates), 3) if wrates else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--io-samples", default="")
    ap.add_argument("--profile-json", default="")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    if not os.path.isfile(args.sqlite):
        print(f"[offload-summary] sqlite not found: {args.sqlite}", file=sys.stderr)
        return 1

    db = sqlite3.connect(args.sqlite)
    cur = db.cursor()
    summary = {
        "capture": "cudaProfilerApi measured steps; nsys --trace=cuda,nvtx,osrt",
        "gpu": gpu_totals(cur),
        "nvtx": nvtx_totals(cur),
        "nvme": io_totals(args.io_samples),
    }
    db.close()

    # Headline ratio: how much of GPU wall is stall (proxy for NVMe swap-in wait).
    span = summary["gpu"].get("gpu_span_ms", 0.0)
    if span > 0:
        summary["gpu"]["gpu_idle_frac"] = round(summary["gpu"]["gpu_idle_ms"] / span, 3)

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[offload-summary] wrote {args.output_json}")

    # Fold into profile.json when it is small enough to rewrite cheaply.
    pj = args.profile_json
    if pj and os.path.isfile(pj):
        size_mb = os.path.getsize(pj) / 1e6
        if size_mb <= 60:
            try:
                with open(pj) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data["offload_io"] = summary
                    with open(pj, "w") as f:
                        json.dump(data, f)
                    print(f"[offload-summary] folded offload_io into {pj}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"[offload-summary] could not fold into {pj}: {e}", file=sys.stderr)
        else:
            print(f"[offload-summary] {pj} is {size_mb:.0f}MB (>60MB); "
                  f"see sibling {os.path.basename(args.output_json)} instead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
