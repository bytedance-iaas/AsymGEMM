#!/usr/bin/env python3
"""Decompose the sTP-MoE backward from an nsys trace.sqlite: per-device kernel busy vs
idle, memcpy class volumes/times, top kernels, and the largest gaps (stall receipt).
Usage: analyze_stp_bwd.py trace.sqlite"""
import sqlite3
import sys


def main() -> int:
    db = sqlite3.connect(sys.argv[1])
    q = lambda s: db.execute(s).fetchall()

    # analysis window: the span of all kernels after warmup settles (last 60% of trace)
    (t0, t1), = q("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    w0 = t0 + int((t1 - t0) * 0.4)
    win = f"start > {w0}"

    print(f"window: {(t1-w0)/1e9:.1f}s of trace")
    for dev, in q("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        (busy,), = q(f"SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId={dev} AND {win}")
        print(f"dev{dev}: kernel busy {busy/1e9:.1f}s ({100*busy/(t1-w0):.0f}% of window)")
    rows = q(f"""SELECT copyKind, SUM(end-start)/1e9, SUM(bytes)/1e9, COUNT(*)
                 FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE {win} GROUP BY copyKind""")
    kinds = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P?"}
    for kind, secs, gb, n in rows:
        print(f"memcpy {kinds.get(kind, kind)}: {secs:.1f}s busy, {gb:.1f} GB, {n} copies")
    rows = q(f"""SELECT names.value, SUM(k.end-k.start)/1e9 AS s, COUNT(*)
                 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds names ON k.shortName = names.id
                 WHERE {win} GROUP BY names.value ORDER BY s DESC LIMIT 8""")
    print("top kernels:")
    for name, secs, n in rows:
        print(f"  {secs:7.1f}s  n={n:<6} {name[:80]}")
    # biggest inter-kernel gaps per device (stall receipt)
    for dev in (0, 1):
        rows = q(f"""SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
                     WHERE deviceId={dev} AND {win} ORDER BY start""")
        if not rows:
            continue
        gaps = []
        prev_end = rows[0][1]
        for s, e in rows[1:]:
            if s > prev_end:
                gaps.append(s - prev_end)
            prev_end = max(prev_end, e)
        gaps.sort(reverse=True)
        total_gap = sum(gaps) / 1e9
        print(f"dev{dev}: total idle-gap {total_gap:.1f}s; top gaps(ms): {[round(g/1e6,1) for g in gaps[:6]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
