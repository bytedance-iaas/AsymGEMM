#!/usr/bin/env python3
"""sEP E3 isolated probe (gb200_ep.md G-E3.1 + probe-level G-E1.3/G-E2.2).

Real grouped GEMM kernels, q3-30b-a3b expert geometry (E=128, N=768, K=2048), synthetic
hot-expert skew alpha; three modes on the 2-GPU pair:
  static    : E/2 ownership per device (vanilla EP)          -> measures the assignment floor
  hostsplit : exact-optimal per-launch LPT split (may split an expert via duplicated ids)
  queue     : the ep_queued kernel — shared pinned counters, front/back sides

Metrics (the gate numbers): per-device busy ms, imbalance %, wall ms, static/queue ratio,
hostsplit replan ms, balanced-routing overhead, and NUMERICS (queue union vs single-device
reference: bitwise; reference vs torch.matmul: bf16 band).

Standalone-allowed (HC3): pinned allocations are small test buffers (<1 GB).
Usage: .venv/bin/python scripts/testing/ep_queue_probe.py [--m-total 262144]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

E, N, K = 128, 768, 2048
BLOCK_M = 64  # DG_BF16_BLOCK_M default in the launcher


def build_counts(m_total: int, alpha: float, generator: torch.Generator) -> list[int]:
    """Expert token counts: expert 0 gets alpha share; rest ~uniform. BLOCK_M-aligned."""
    counts = [0] * E
    hot = int(m_total * alpha) // BLOCK_M * BLOCK_M
    counts[0] = hot
    remaining = m_total - hot
    base = remaining // (E - 1) // BLOCK_M * BLOCK_M
    for i in range(1, E):
        counts[i] = base
    leftover = m_total - sum(counts)
    counts[1] += leftover // BLOCK_M * BLOCK_M
    return counts


def build_meta(counts: list[int], expert_ids: list[int], device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """offsets int32 pairs [start,end]*G over the CONTIGUOUS global row layout, experts ids + -1."""
    starts = []
    acc = 0
    global_start = {}
    for e in range(E):
        global_start[e] = acc
        acc += counts[e]
    pairs, ids = [], []
    for e in expert_ids:
        if counts[e] == 0:
            continue
        pairs += [global_start[e], global_start[e] + counts[e]]
        ids.append(e)
    ids.append(-1)
    offsets = torch.tensor(pairs, dtype=torch.int32, device=device)
    experts = torch.tensor(ids, dtype=torch.int32, device=device)
    return offsets, experts, len(ids)  # list_size = groups + 1


def build_seg_meta(segments: list[tuple[int, int, int]], device):
    pairs, ids = [], []
    for expert, start, end in segments:
        if end > start:
            pairs += [start, end]
            ids.append(expert)
    ids.append(-1)
    return (torch.tensor(pairs, dtype=torch.int32, device=device),
            torch.tensor(ids, dtype=torch.int32, device=device), len(ids))


def build_split_meta(counts: list[int], assignment: list[list[tuple[int, int, int]]], device_list):
    """assignment[dev] = list of (expert, row_start, row_end) GLOBAL rows (may split experts)."""
    metas = []
    for dev, items in zip(device_list, assignment):
        pairs, ids = [], []
        for expert, start, end in items:
            if end > start:
                pairs += [start, end]
                ids.append(expert)
        ids.append(-1)
        metas.append((torch.tensor(pairs, dtype=torch.int32, device=dev),
                      torch.tensor(ids, dtype=torch.int32, device=dev), len(ids)))
    return metas


import os
HOT_CHUNK_ROWS = int(os.environ.get("EP_HOT_CHUNK_ROWS", "8192"))  # hot-expert chunk size:
# finer -> better balance quantum; coarser -> fewer host-B re-fetches (~25us each; host TMA
# reads do NOT L2-cache). Wall-minimizing point measured by sweep.
HOT_FANOUT_CAP = 8      # fine-chunk at most this many experts (distinct slices must fit L2)


def chunk_segments(counts: list[int], chunk_rows: int) -> list[tuple[int, int, int]]:
    """(expert, m-sub-range) work segments, expert-sorted (front/back affinity order).
    Per-expert adaptive granularity (measured physics, 2026-07-06):
      - average-size experts stay WHOLE (each chunk re-streams the expert's full B slice
        from host ~25us — fine-chunking ALL experts thrashes L2 and taxes balanced routing)
      - the few HOT experts (> 2x average) chunk at HOT_CHUNK_ROWS: one hot slice stays
        L2-resident across its chunks, so fine granularity there is ~free and gives the
        queue its balance quantum exactly where imbalance lives."""
    avg = max(1, sum(counts) // max(1, sum(1 for c in counts if c > 0)))
    hot_ids = sorted((e for e in range(E) if counts[e] > 2 * avg),
                     key=lambda e: -counts[e])[:HOT_FANOUT_CAP]
    segs = []
    acc = 0
    for e in range(E):
        c = counts[e]
        start = acc
        acc += c
        step = HOT_CHUNK_ROWS if e in hot_ids else max(chunk_rows, c)
        while c > 0:
            take = min(c, step)
            segs.append((e, start, start + take))
            start += take
            c -= take
    return segs


SEG_FIXED_COST_ROWS = 2000  # measured ~25us/segment host-B refetch ~= 2000 rows of m-loop


def lpt_split(segments: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Exact-per-launch scheduler baseline: LPT over the SAME chunked segments. Cost model
    = rows + fixed per-segment B-refetch (measured), else tiny segments misbalance busy."""
    def cost(seg):
        return (seg[2] - seg[1]) + SEG_FIXED_COST_ROWS
    order = sorted(segments, key=lambda seg: -cost(seg))
    loads = [0, 0]
    out: list[list[tuple[int, int, int]]] = [[], []]
    for seg in order:
        light = 0 if loads[0] <= loads[1] else 1
        out[light].append(seg)
        loads[light] += cost(seg)
    return out


def run_mode(mode, a, b_pinned, d_bufs, metas, queue, devices, streams):
    import asym_gemm
    events = []
    for d in d_bufs:
        d.zero_()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    if queue is not None:
        queue.zero_()
    t0 = time.perf_counter()
    for slot, dev in enumerate(devices):
        with torch.cuda.device(dev), torch.cuda.stream(streams[slot]):
            start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
            start_ev.record()
            offsets, experts, list_size = metas[slot]
            if mode == "queue":
                asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(
                    a[slot], b_pinned, d_bufs[slot], offsets, experts, list_size,
                    queue, slot, "nk", False)
            else:
                asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                    a[slot], b_pinned, d_bufs[slot], offsets, experts, list_size, "nk", False)
            end_ev.record()
            events.append((start_ev, end_ev))
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    wall_ms = (time.perf_counter() - t0) * 1e3
    busy = [s.elapsed_time(e) for s, e in events]
    return busy, wall_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-total", type=int, default=262144)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=0,
                        help="0 = adaptive: max(BLOCK_M, M/128) — pins balance quantum ~2% and "
                             "amortizes the per-segment host-B refetch as M grows")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    import asym_gemm  # noqa: F401

    gen = torch.Generator().manual_seed(0)
    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    streams = [torch.cuda.Stream(device=d) for d in devices]
    m_total = args.m_total

    # operands: A replicated (both devices), B ONE pinned bank shared, D per device
    a_host = (torch.randn(m_total, K, generator=gen, dtype=torch.float32) / 8).to(torch.bfloat16)
    a = [a_host.to(d) for d in devices]
    b_pinned = ((torch.randn(E, N, K, generator=gen, dtype=torch.float32) / 32).to(torch.bfloat16)).pin_memory()
    d_bufs = [torch.zeros(m_total, N, dtype=torch.bfloat16, device=d) for d in devices]
    queue = torch.zeros(3, dtype=torch.int32).pin_memory()

    if args.chunk_rows <= 0:
        # Two constraints (measured 2026-07-06, q3-30b-a3b geometry, membind=0,1):
        #   B-refetch tax  ~= 13900/chunk_rows %  (per-segment host-B re-stream ~25us; M-indep)
        #   balance quantum ~= 200*chunk_rows/M %  (serial m-loop ~11.5us/m-block)
        # Both <=2% requires chunk >= ~8192 AND M >= ~1M rows (production per-layer routed M).
        # Must EXCEED the balanced per-expert size (m/E) or every expert splits into
        # 2 chunks (one tiny) and the B-refetch tax doubles at balanced routing.
        args.chunk_rows = (max(BLOCK_M, 8192, m_total // E + BLOCK_M) + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        print(f"[probe] adaptive chunk_rows = {args.chunk_rows} "
              f"(expected tax ~{13900/args.chunk_rows:.1f}%, quantum ~{200*args.chunk_rows/m_total:.1f}%)")
    report = {"m_total": m_total, "chunk_rows": args.chunk_rows, "geometry": {"E": E, "N": N, "K": K}, "alphas": {}}
    failures = []

    # ---------- numerics gate first (small, alpha=0.5) ----------
    counts = build_counts(m_total, 0.5, gen)
    full_meta0 = build_meta(counts, list(range(E)), devices[0])
    # reference: ALL experts on dev0
    ref = torch.zeros_like(d_bufs[0])
    import asym_gemm as ag
    with torch.cuda.device(devices[0]):
        ag.m_grouped_bf16_asym_gemm_nt_contiguous(a[0], b_pinned, ref, *full_meta0, "nk", False)
    torch.cuda.synchronize(0)
    # torch reference band check on a slice (expert 0 rows)
    rows = slice(0, min(counts[0], 4096))
    torch_ref = (a[0][rows].float() @ b_pinned[0].to(devices[0]).float().T).to(torch.bfloat16)
    band = (ref[rows].float() - torch_ref.float()).abs().max().item()
    print(f"[probe] static-vs-torch band (expert0 slice): {band:.4f}")
    if band > 0.35:
        failures.append(f"static kernel vs torch band {band}")
    # queue union == reference bitwise (chunked item space)
    segs = chunk_segments(counts, args.chunk_rows)
    metas_q = [build_seg_meta(segs, devices[0]), build_seg_meta(segs, devices[1])]
    busy, _ = run_mode("queue", a, b_pinned, d_bufs, metas_q, queue, devices, streams)
    n_items = (len(segs)) * (N // 64)
    print(f"[probe] queue counters after run: {queue.tolist()} (segments={len(segs)}, items={n_items})")
    union = d_bufs[0] + d_bufs[1].to(devices[0])
    bit_ok = torch.equal(union, ref)
    overlap = ((d_bufs[0] != 0) & (d_bufs[1].to(devices[0]) != 0)).any().item()
    print(f"[probe] queue union bitwise==static-ref: {bit_ok}; row-overlap(double-compute): {overlap}")
    if not bit_ok:
        failures.append("queue union != static reference (bitwise)")
    if overlap:
        failures.append("queue double-computed rows")

    # ---------- balance/perf sweep ----------
    for alpha in (0.0, 0.25, 0.50, 0.75):
        counts = build_counts(m_total, alpha, gen)
        entry = {}
        # static: E/2 ownership
        metas_static = build_split_meta(
            counts,
            [[(e, sum(counts[:e]), sum(counts[:e + 1])) for e in range(0, E // 2)],
             [(e, sum(counts[:e]), sum(counts[:e + 1])) for e in range(E // 2, E)]],
            devices)
        # hostsplit: chunk + LPT (timed = the replan cost EG3(i) charges)
        t0 = time.perf_counter()
        segs = chunk_segments(counts, args.chunk_rows)
        assignment = lpt_split(segs)
        replan_ms = (time.perf_counter() - t0) * 1e3
        metas_host = [build_seg_meta(assignment[0], devices[0]),
                      build_seg_meta(assignment[1], devices[1])]
        metas_queue = [build_seg_meta(segs, devices[0]), build_seg_meta(segs, devices[1])]

        for mode, metas in (("static", metas_static), ("hostsplit", metas_host), ("queue", metas_queue)):
            best = None
            for _ in range(args.iters):
                busy, wall = run_mode(mode, a, b_pinned, d_bufs, metas,
                                      queue if mode == "queue" else None, devices, streams)
                if best is None or wall < best[1]:
                    best = (busy, wall)
            busy, wall = best
            imb = abs(busy[0] - busy[1]) / max(max(busy), 1e-9) * 100
            entry[mode] = {"busy_ms": [round(x, 2) for x in busy], "imb_pct": round(imb, 1),
                           "wall_ms": round(wall, 2)}
            if mode == "hostsplit":
                entry[mode]["replan_ms"] = round(replan_ms, 3)
        entry["static_over_queue"] = round(max(entry["static"]["busy_ms"]) / max(entry["queue"]["busy_ms"]), 3)
        report["alphas"][str(alpha)] = entry
        print(f"[probe] alpha={alpha}: static busy={entry['static']['busy_ms']} imb={entry['static']['imb_pct']}% | "
              f"hostsplit imb={entry['hostsplit']['imb_pct']}% replan={entry['hostsplit'].get('replan_ms')}ms | "
              f"queue busy={entry['queue']['busy_ms']} imb={entry['queue']['imb_pct']}% | "
              f"static/queue={entry['static_over_queue']}x")

    # ---------- gates ----------
    for alpha in ("0.25", "0.5", "0.75"):
        key = alpha if alpha in report["alphas"] else str(float(alpha))
        e = report["alphas"].get(key)
        if e is None:
            continue
        if e["queue"]["imb_pct"] > 5.0:
            failures.append(f"queue imbalance {e['queue']['imb_pct']}% at alpha={key}")
        expected = 1 + float(key) * 0.9  # (1+alpha) with 10% slack
        if e["static_over_queue"] < expected * 0.8:
            failures.append(f"static/queue {e['static_over_queue']} < expected ~{1+float(key):.2f} at alpha={key}")
    bal = report["alphas"]["0.0"]
    overhead = max(bal["queue"]["busy_ms"]) / max(max(bal["static"]["busy_ms"]), 1e-9)
    report["balanced_overhead_ratio"] = round(overhead, 4)
    print(f"[probe] balanced-routing queue/static overhead: {overhead:.4f} (gate <= 1.02)")
    if overhead > 1.02:
        failures.append(f"balanced overhead {overhead:.4f} > 1.02")

    report["failures"] = failures
    report["pass"] = not failures
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    print(f"[probe] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
