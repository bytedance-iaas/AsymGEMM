#!/usr/bin/env python3
"""GH200-96GB HBM occupier (standardize_tps_96gb.md).

Pins ONE simulated GPU down to a 96-GB card's visible HBM: allocates a
single uint8 tensor sized (current_free - TARGET_FREE) and sleeps forever.
Target-free sizing auto-compensates foreign residents. Run one per
simulated GPU, INSIDE the container, with CUDA_VISIBLE_DEVICES set to
that GPU alone.

Usage: python hbm96_occupy.py [--target-free-gib 95.6]
Prints one status line then sleeps; SIGTERM/SIGINT to release.
"""
import argparse
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-free-gib", type=float, default=95.6,
                    help="HBM to leave free (GH200-96GB visible = 95.6 GiB)")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA device visible"
    dev = 0  # CVD restricts to the one simulated GPU
    free_b, total_b = torch.cuda.mem_get_info(dev)
    target_free = int(args.target_free_gib * 2**30)
    occupy = free_b - target_free
    if occupy <= 0:
        print(f"[occupier] nothing to occupy: free {free_b/2**30:.1f} GiB "
              f"<= target {args.target_free_gib} GiB", flush=True)
    else:
        t = torch.empty(occupy, dtype=torch.uint8, device=f"cuda:{dev}")
        t.fill_(0)  # touch pages so the allocation is real
        free2, _ = torch.cuda.mem_get_info(dev)
        print(f"[occupier] pid={os.getpid()} occupied {occupy/2**30:.1f} GiB, "
              f"free now {free2/2**30:.1f} GiB (target {args.target_free_gib}) "
              f"total {total_b/2**30:.1f} GiB", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
