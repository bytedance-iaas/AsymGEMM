#!/usr/bin/env python3
"""Extract per-lane C2C/NVLink evidence from an nsys sqlite export -> lane_bw.json.

gb200_tp.md I1 tooling; consumed by the I3+ gates ("both lanes active in base-GEMM
windows", P6 ">=170 GB/s in streamed windows", exchange overlap receipts).

What it reports (best-effort per table availability in the nsys version):
  memcpy: per-device H2D/D2H copy bytes, busy time, effective GB/s p50/p95
          (act offload / restage / P2P exchanges — the copy-engine traffic).
  kernels: per-device total/asym-GEMM kernel busy time + wall span + overlap fraction
          (the weight-STREAMING traffic is in-kernel TMA — it shows up as GEMM kernel
          time, NOT as memcpys; lane delivery for it = weight-bytes / kernel-time,
          computed by the caller who knows the weight bytes).
  nvtx:   spans matching --window-regex (default: asym/stp GEMM ranges) so callers can
          window the classes above.

Usage: extract_lane_bw.py --sqlite trace.sqlite --out lane_bw.json [--window-regex asym]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


def _tables(cur) -> set[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in cur.fetchall()}


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-regex", default=r"asym|stp|base_forward|base_dx")
    args = parser.parse_args()

    conn = sqlite3.connect(args.sqlite)
    cur = conn.cursor()
    tables = _tables(cur)
    report: dict = {"sqlite": args.sqlite, "tables_seen": sorted(t for t in tables if "CUPTI" in t or "NVTX" in t or "GPU_METRICS" in t)}

    # ---- memcpys ----
    memcpy_table = next((t for t in ("CUPTI_ACTIVITY_KIND_MEMCPY",) if t in tables), None)
    if memcpy_table:
        cur.execute(
            f"SELECT deviceId, copyKind, bytes, start, end, streamId FROM {memcpy_table}"
        )
        per_class: dict[tuple[int, str], dict] = defaultdict(lambda: {"bytes": 0, "busy_ns": 0, "rates": []})
        kind_names = {1: "h2d", 2: "d2h", 8: "d2d", 10: "p2p", 11: "p2p"}
        for device_id, copy_kind, nbytes, start, end, stream_id in cur.fetchall():
            label = kind_names.get(copy_kind, f"kind{copy_kind}")
            rec = per_class[(device_id, label)]
            rec["bytes"] += nbytes or 0
            duration = max((end or 0) - (start or 0), 1)
            rec["busy_ns"] += duration
            if nbytes:
                rec["rates"].append(nbytes / duration)  # bytes/ns == GB/s
        report["memcpy"] = {}
        for (device_id, label), rec in sorted(per_class.items()):
            rates = sorted(rec["rates"])
            report["memcpy"][f"dev{device_id}_{label}"] = {
                "bytes": rec["bytes"],
                "busy_ms": round(rec["busy_ns"] / 1e6, 2),
                "avg_GBps": round(rec["bytes"] / rec["busy_ns"], 1) if rec["busy_ns"] else None,
                "p50_GBps": round(_percentile(rates, 0.5), 1) if rates else None,
                "p95_GBps": round(_percentile(rates, 0.95), 1) if rates else None,
                "count": len(rec["rates"]),
            }

    # ---- kernels ----
    kernel_table = next(
        (t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if t in tables),
        None,
    )
    if kernel_table:
        has_strings = "StringIds" in tables
        name_join = "LEFT JOIN StringIds s ON k.shortName = s.id" if has_strings else ""
        name_expr = "s.value" if has_strings else "''"
        cur.execute(
            f"SELECT k.deviceId, {name_expr}, k.start, k.end FROM {kernel_table} k {name_join}"
        )
        per_dev: dict[int, dict] = defaultdict(
            lambda: {"kernel_busy_ns": 0, "asym_busy_ns": 0, "span": [None, None], "asym_windows": []}
        )
        gemm_re = re.compile(r"asym|gemm|sm100", re.IGNORECASE)
        for device_id, name, start, end in cur.fetchall():
            rec = per_dev[device_id]
            duration = max((end or 0) - (start or 0), 0)
            rec["kernel_busy_ns"] += duration
            span = rec["span"]
            span[0] = start if span[0] is None else min(span[0], start)
            span[1] = end if span[1] is None else max(span[1], end)
            if name and gemm_re.search(name):
                rec["asym_busy_ns"] += duration
                rec["asym_windows"].append((start, end))
        report["kernels"] = {}
        for device_id, rec in sorted(per_dev.items()):
            wall = (rec["span"][1] - rec["span"][0]) if rec["span"][0] is not None else 0
            report["kernels"][f"dev{device_id}"] = {
                "kernel_busy_ms": round(rec["kernel_busy_ns"] / 1e6, 1),
                "asym_gemm_busy_ms": round(rec["asym_busy_ns"] / 1e6, 1),
                "wall_span_ms": round(wall / 1e6, 1),
                "asym_windows": len(rec["asym_windows"]),
            }
        # cross-device GEMM overlap: intersection of asym windows dev0 x dev1
        devs = sorted(per_dev)
        if len(devs) >= 2:
            def merge(windows):
                out = []
                for s, e in sorted(windows):
                    if out and s <= out[-1][1]:
                        out[-1][1] = max(out[-1][1], e)
                    else:
                        out.append([s, e])
                return out

            w0, w1 = merge(per_dev[devs[0]]["asym_windows"]), merge(per_dev[devs[1]]["asym_windows"])
            i = j = inter = 0
            while i < len(w0) and j < len(w1):
                lo = max(w0[i][0], w1[j][0]); hi = min(w0[i][1], w1[j][1])
                if hi > lo:
                    inter += hi - lo
                if w0[i][1] < w1[j][1]:
                    i += 1
                else:
                    j += 1
            total0 = sum(e - s for s, e in w0)
            report["kernels"]["asym_gemm_overlap_frac_of_dev0"] = round(inter / total0, 3) if total0 else None

    # ---- nvtx windows ----
    nvtx_table = next((t for t in ("NVTX_EVENTS",) if t in tables), None)
    if nvtx_table:
        cur.execute(f"SELECT text, start, end FROM {nvtx_table} WHERE text IS NOT NULL")
        window_re = re.compile(args.window_regex, re.IGNORECASE)
        spans = [(t, s, e) for t, s, e in cur.fetchall() if t and window_re.search(t) and e]
        report["nvtx_windows"] = {"count": len(spans), "regex": args.window_regex,
                                  "total_ms": round(sum(e - s for _, s, e in spans) / 1e6, 1)}

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "tables_seen"}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
