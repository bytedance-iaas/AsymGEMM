#!/usr/bin/env python3
"""hbm96_occupy.py — GH200-96GB HBM simulator (standardize_tps_96gb.md).

One process per simulated GPU: allocates a single uint8 tensor sized
(current_free − target) on cuda:0 of its restricted view and sleeps.
Target-free sizing auto-compensates foreign residents. Run it inside the
container with the GPU restricted (NVIDIA_VISIBLE_DEVICES=<phys id>):
    .venv/bin/python agent/anchors_tmp/hbm96_occupy.py [--target-gib 95.6]
Prints one OCCUPIER-READY line, then sleeps forever (kill -9 to release).
"""
import argparse, os, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--target-gib", type=float, default=95.6)
a = ap.parse_args()
torch.cuda.init()
free, total = torch.cuda.mem_get_info(0)
tgt = int(a.target_gib * (1 << 30))
occ = free - tgt
assert occ > 0, f"free {free/2**30:.1f}GiB already <= target {a.target_gib}GiB"
buf = torch.empty(occ, dtype=torch.uint8, device="cuda:0")
free2, _ = torch.cuda.mem_get_info(0)
print(f"OCCUPIER-READY pid={os.getpid()} phys={os.environ.get('NVIDIA_VISIBLE_DEVICES','?')} "
      f"occupied={occ/2**30:.2f}GiB free_after={free2/2**30:.2f}GiB", flush=True)
while True:
    time.sleep(3600)
