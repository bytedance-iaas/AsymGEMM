#!/usr/bin/env python
"""Replay a torch CUDA memory snapshot to its peak live-set and attribute by allocation frame."""
import pickle, sys, collections
from pathlib import Path

_TORCH = ("site-packages/torch/", "/torch/cuda/", "/torch/_", "/torch/autograd/",
          "/torch/nn/modules/module.py", "c10/")


def uf(fr):
    fr = fr or []
    for f in fr:
        fn = str(f.get("filename", ""))
        if any(s in fn for s in ("asym_gemm/", "transformers/models/", "llamafactory/")):
            return f"{Path(fn).name}:{f.get('line','?')}:{f.get('name','?')}"
    for f in fr:
        fn = str(f.get("filename", ""))
        if not any(m in fn for m in _TORCH):
            return f"{Path(fn).name}:{f.get('line','?')}:{f.get('name','?')}"
    return fr[0].get("name", "?") if fr else "<none>"


def main(path):
    snap = pickle.load(open(path, "rb"))
    best = None
    for dev in snap.get("device_traces") or []:
        live = {}; tot = 0; peak = 0; pl = None
        for ev in dev:
            a = ev.get("action"); addr = ev.get("addr"); sz = int(ev.get("size") or 0)
            if a == "alloc":
                live[addr] = (sz, ev.get("frames")); tot += sz
                if tot > peak: peak = tot; pl = dict(live)
            elif a in ("free_completed", "free_requested"):
                if addr in live: tot -= live[addr][0]; del live[addr]
        if pl and (best is None or peak > best[0]): best = (peak, pl)
    if best is None:
        print("no device_traces / no peak found"); return
    peak, live = best
    byf = collections.defaultdict(int)
    for _, (sz, fr) in live.items(): byf[uf(fr)] += sz
    print(f"PEAK {peak/2**20:,.0f} MiB across {len(live)} blocks")
    for f, b in sorted(byf.items(), key=lambda x: -x[1])[:12]:
        if b / 2**20 >= 40: print(f"  {b/2**20:9,.0f} MiB  {f}")


if __name__ == "__main__":
    main(sys.argv[1])
