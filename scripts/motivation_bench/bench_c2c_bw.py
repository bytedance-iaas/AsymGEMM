#!/usr/bin/env python3
"""GB200 C2C bandwidth sweep — the SuperOffload Fig.7 counterpart.

Pinned-host <-> HBM transfer bandwidth vs tensor size (0.25 MB -> 1 GB,
doubling), both directions, copy-engine path (torch copy_ on pinned buffers).
House micro protocol: CUDA events, warmup + timed iters per size, 3 runs
(first discarded). JSON -> profiling_results/motivation/c2c_bw.json.
Run in-container on an idle GPU, membind = the GPU's NUMA node.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/c2c_bw.json"
SIZES_MB = [0.25 * (2 ** i) for i in range(13)]  # 0.25 MB .. 1024 MB
RUNS = 3  # first is warmup, discarded


def _iters_for(mb: float) -> tuple[int, int]:
    if mb <= 4:
        return 20, 200
    if mb <= 64:
        return 10, 100
    return 5, 30


def _time_copies(dst: torch.Tensor, src: torch.Tensor, warmup: int, timed: int) -> float:
    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed):
        dst.copy_(src, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / timed  # ms per copy


def bench_size(mb: float) -> dict:
    n = int(mb * 1024 * 1024) // 2  # bf16 elements
    host = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    host.normal_()
    dev = host.to("cuda", non_blocking=False)
    warmup, timed = _iters_for(mb)
    res = {"mb": mb, "runs": []}
    for _ in range(RUNS):
        h2d_ms = _time_copies(dev, host, warmup, timed)
        d2h_ms = _time_copies(host, dev, warmup, timed)
        gb = mb * 1024 * 1024 / 1e9  # decimal GB (house/SuperOffload axis)
        res["runs"].append({"h2d_gbs": gb / (h2d_ms / 1000.0),
                            "d2h_gbs": gb / (d2h_ms / 1000.0)})
    del host, dev
    torch.cuda.empty_cache()
    return res


def main() -> None:
    torch.manual_seed(0)
    out = {
        "spec": {
            "what": "pinned-host <-> HBM copy bandwidth vs size (copy_ engine path)",
            "device": torch.cuda.get_device_name(0),
            "sizes_mb": SIZES_MB,
            "dtype": "bf16",
            "units": "decimal GB/s (bytes/1e9)",
            "runs": RUNS,
            "numactl": os.environ.get("NUMACTL_MEMBIND", "<none>"),
        },
        "sizes": [bench_size(mb) for mb in SIZES_MB],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    for s in out["sizes"]:
        runs = s["runs"][1:]
        h2d = sum(r["h2d_gbs"] for r in runs) / len(runs)
        d2h = sum(r["d2h_gbs"] for r in runs) / len(runs)
        print(f"{s['mb']:>8.2f} MB  CPU->GPU {h2d:7.1f} GB/s   GPU->CPU {d2h:7.1f} GB/s")


if __name__ == "__main__":
    main()
