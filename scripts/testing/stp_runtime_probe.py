#!/usr/bin/env python3
"""Stage I1 isolated probe (gb200_tp.md): topology + lanes + allreduce2 + a REAL dev1 GEMM.

Standalone-allowed per HC3: pinned host allocations are capped to SMALL test buffers
(<= ~4 GiB), never a model arena. PASS bar:
  - peer access both ways; a real asym GEMM launches on dev1 (JIT FIX A works);
  - P2P copy >= 700 GB/s/dir sustained, both dirs concurrently;
  - both C2C lanes pull the SAME pinned buffer concurrently at high BW (shared-arena legality);
  - allreduce2 of [200000, 8192] bf16 (~3.05 GiB/device) < 8 ms;
  - per-kernel enqueue cost from ONE host thread < 30 us.

Usage: .venv/bin/python scripts/testing/stp_runtime_probe.py --pair 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("ASYM_STP", "1")
os.environ.setdefault("ASYM_STP_TP_SIZE", "2")

import torch  # noqa: E402

GIB = 1024**3


def _wall_time_streams(fn, iters: int, devices) -> float:
    """Wall seconds per iter for async work launched by fn(), synced on all devices."""
    for _ in range(3):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    return (time.perf_counter() - start) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="0,1")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    pair = tuple(int(x) for x in args.pair.split(","))
    assert len(pair) == 2

    report: dict = {"pair": pair, "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")}
    failures: list[str] = []

    from asym_gemm.training.stp_runtime import STPRuntime

    t0 = time.perf_counter()
    rt = STPRuntime(dev_ids=pair)
    report["init_s"] = round(time.perf_counter() - t0, 3)
    print(f"[probe] STPRuntime init (peer enable + JIT prewarm BOTH devices): {report['init_s']}s")

    # ---- dev1 REAL GEMM numerics (the FIX A acceptance) ----
    from asym_gemm.training.frozen_linear import _asym_bf16_nt

    for slot, dev in enumerate(rt.d):
        with torch.cuda.device(dev):
            a = torch.randn(4096, 8192, device=dev, dtype=torch.bfloat16)
            b_cpu = (torch.randn(4096, 8192, dtype=torch.bfloat16) / 64.0).pin_memory()
            got = _asym_bf16_nt(a, b_cpu)
            ref = a @ b_cpu.to(dev).T
            torch.cuda.synchronize(dev)
            rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
            report[f"gemm_dev{slot}_max_rel_err"] = rel
            print(f"[probe] asym GEMM on {dev}: max rel err vs torch = {rel:.3e}")
            if rel > 3e-2:
                failures.append(f"gemm numerics dev{slot} rel={rel}")

    # ---- P2P bandwidth, each direction + full duplex ----
    n_bytes = 4 * GIB
    x0 = torch.empty(n_bytes // 2, dtype=torch.bfloat16, device=rt.d[0])
    x1 = torch.empty(n_bytes // 2, dtype=torch.bfloat16, device=rt.d[1])

    sec = _wall_time_streams(lambda: rt.bcast01(x0), 5, rt.d)
    report["p2p_0to1_GBps"] = round(n_bytes / sec / 1e9, 1)
    sec = _wall_time_streams(lambda: rt.to0(x1), 5, rt.d)
    report["p2p_1to0_GBps"] = round(n_bytes / sec / 1e9, 1)

    # Raw concurrent copies on the two p2p streams (composing the ambient-exit primitives
    # back-to-back would serialize via the entry-on-ambient contract — see stp_runtime._p2p_copy).
    dup0 = torch.empty_like(x0, device=rt.d[1])
    dup1 = torch.empty_like(x1, device=rt.d[0])

    def duplex():
        with torch.cuda.stream(rt.p2p[0]):
            dup0.copy_(x0, non_blocking=True)
        with torch.cuda.stream(rt.p2p[1]):
            dup1.copy_(x1, non_blocking=True)

    sec = _wall_time_streams(duplex, 5, rt.d)
    report["p2p_duplex_GBps_per_dir"] = round(n_bytes / sec / 1e9, 1)
    print(
        f"[probe] P2P GB/s: 0->1 {report['p2p_0to1_GBps']}, 1->0 {report['p2p_1to0_GBps']},"
        f" duplex/dir {report['p2p_duplex_GBps_per_dir']}"
    )
    if report["p2p_duplex_GBps_per_dir"] < 700:
        failures.append(f"p2p duplex {report['p2p_duplex_GBps_per_dir']} GB/s < 700")

    # ---- both C2C lanes pull the SAME pinned buffer concurrently (shared-arena legality) ----
    pinned = torch.empty(2 * GIB, dtype=torch.uint8).pin_memory()
    dst = [torch.empty(2 * GIB, dtype=torch.uint8, device=d) for d in rt.d]

    def one_lane(slot: int):
        with torch.cuda.device(rt.d[slot]), torch.cuda.stream(rt.h2d[slot]):
            dst[slot].copy_(pinned, non_blocking=True)

    sec = _wall_time_streams(lambda: one_lane(0), 5, [rt.d[0]])
    report["h2d_lane0_solo_GBps"] = round(2 * GIB / sec / 1e9, 1)
    sec = _wall_time_streams(lambda: one_lane(1), 5, [rt.d[1]])
    report["h2d_lane1_solo_GBps"] = round(2 * GIB / sec / 1e9, 1)

    def both_lanes():
        one_lane(0)
        one_lane(1)

    sec = _wall_time_streams(both_lanes, 5, rt.d)
    report["h2d_both_lanes_shared_pinned_GBps_per_lane"] = round(2 * GIB / sec / 1e9, 1)
    print(
        f"[probe] H2D GB/s: lane0 solo {report['h2d_lane0_solo_GBps']}, lane1 solo"
        f" {report['h2d_lane1_solo_GBps']}, both-from-SAME-pinned {report['h2d_both_lanes_shared_pinned_GBps_per_lane']}/lane"
    )

    # ---- allreduce2 of [200000, 8192] bf16 (~3.05 GiB per device) ----
    y0 = torch.randn(200000, 8192, device=rt.d[0], dtype=torch.bfloat16)
    y1 = torch.randn(200000, 8192, device=rt.d[1], dtype=torch.bfloat16)
    # numerics once: y0+y1 on both sides
    r0, r1 = y0.clone(), y1.clone()
    expected = (y0.float().cpu() + y1.float().cpu().to("cpu"))
    rt.allreduce2(r0, r1)
    torch.cuda.synchronize(rt.d[0]); torch.cuda.synchronize(rt.d[1])
    e0 = (r0.float().cpu() - expected).abs().max().item()
    e1 = (r1.float().cpu() - expected).abs().max().item()
    report["allreduce2_max_abs_err"] = max(e0, e1)
    if max(e0, e1) > 0.25:  # bf16 add of ~N(0,1)+N(0,1): representation error only
        failures.append(f"allreduce2 numerics err {max(e0, e1)}")

    sec = _wall_time_streams(lambda: rt.allreduce2(y0, y1), 10, rt.d)
    report["allreduce2_3GiB_ms"] = round(sec * 1e3, 2)
    print(f"[probe] allreduce2([200000,8192] bf16): {report['allreduce2_3GiB_ms']} ms, max abs err {report['allreduce2_max_abs_err']:.3f}")
    if report["allreduce2_3GiB_ms"] >= 8.0:
        failures.append(f"allreduce2 {report['allreduce2_3GiB_ms']} ms >= 8 ms")

    # ---- per-kernel enqueue cost from ONE host thread driving both devices ----
    tiny = [torch.ones(64, device=d) for d in rt.d]
    for d in rt.d:
        torch.cuda.synchronize(d)
    start = time.perf_counter()
    launches = 2000
    for i in range(launches // 2):
        for slot in (0, 1):
            with torch.cuda.device(rt.d[slot]):
                tiny[slot].add_(1.0)
    for d in rt.d:
        torch.cuda.synchronize(d)
    report["enqueue_us_per_kernel"] = round((time.perf_counter() - start) / launches * 1e6, 2)
    print(f"[probe] enqueue cost (1 thread, alternating devices): {report['enqueue_us_per_kernel']} us/kernel")
    if report["enqueue_us_per_kernel"] > 30:
        failures.append(f"enqueue {report['enqueue_us_per_kernel']} us > 30")

    # ---- numa placement of the pinned test buffer ----
    numa_counts = {"N0": 0, "N1": 0}
    try:
        for line in open("/proc/self/numa_maps"):
            for tok in line.split():
                if tok.startswith("N0="):
                    numa_counts["N0"] += int(tok[3:])
                elif tok.startswith("N1="):
                    numa_counts["N1"] += int(tok[3:])
    except OSError:
        pass
    report["numa_pages"] = numa_counts
    print(f"[probe] numa pages: {numa_counts}")

    report["failures"] = failures
    report["pass"] = not failures
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    print(f"[probe] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
