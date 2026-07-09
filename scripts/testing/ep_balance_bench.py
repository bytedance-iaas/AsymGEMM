"""S5a BALANCING microbench (fix_gb200_ep.md): owned-static vs sEP queue on REAL
recorded routing, two processes, real bank shapes, tuned-alpha injection.

Transport is IDENTICAL in both modes (each rank holds the full union A locally; B
streams from the same pinned host bank) so the A/B isolates the ASSIGNMENT POLICY:
  OWNED : rank r executes exactly experts [r*E/2,(r+1)*E/2)'s segments (fixed).
  QUEUE : union list, SHARED system-scope counters, side 0 pops front / 1 back
          (hot-expert chunking at EP_HOT_CHUNK_ROWS, the banked granularity co-design).
Wall per mode = max(rank walls); imbalance = |b0-b1|/max.

Usage:
  .venv/bin/python scripts/testing/ep_balance_bench.py \
      --hist profiling_gb200ep_sg/ep_hist_q3_s20000.json \
      [--m-total 1280000] [--alphas natural,0.25,0.5,0.75] [--layers worst,median]
      [--reps 3] [--out bench.json]
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALIGN = 4096
E, N, K = 128, 768, 2048          # q3-30b gate/up geometry (real shapes)
HOT_CHUNK = int(os.environ.get("EP_HOT_CHUNK_ROWS", "8192"))
HOT_FANOUT_CAP = 8


def shm_path(tag: str) -> str:
    return f"/dev/shm/asym_epbench_{tag}"


def barrier(tag: str, name: str, rank: int, world: int, timeout: float = 900.0) -> None:
    base = shm_path(tag) + f".bar_{name}"
    open(f"{base}.{rank}", "w").close()
    deadline = time.time() + timeout
    while any(not os.path.exists(f"{base}.{r}") for r in range(world)):
        if time.time() > deadline:
            raise RuntimeError(f"barrier {name} timed out (rank {rank})")
        time.sleep(0.01)


def load_layer_counts(hist_path: str, which: list[str]) -> dict[str, list[int]]:
    d = json.load(open(hist_path))
    layers = d["layers"]
    scored = sorted(
        layers.items(),
        key=lambda kv: kv[1].get("static_e2_device_share_max") or 0.0,
    )
    picks: dict[str, list[int]] = {}
    for w in which:
        if w == "worst":
            k, v = scored[-1]
        elif w == "median":
            k, v = scored[len(scored) // 2]
        elif w == "best":
            k, v = scored[0]
        else:
            k, v = w, layers[w]
        picks[f"{w}:{k}"] = v["counts"]
    return picks


def scale_counts(counts: list[int], m_total: int, alpha: float) -> list[int]:
    total = sum(counts) or 1
    scaled = [max(0, round(c * m_total / total)) for c in counts]
    if alpha > 0:  # harness semantic: alpha fraction of ALL rows forced onto expert 0
        moved = 0
        for e in range(1, E):
            take = round(scaled[e] * alpha)
            scaled[e] -= take
            moved += take
        scaled[0] += moved
    drift = m_total - sum(scaled)
    scaled[0] += drift
    return scaled


def chunk_segments(counts: list[int]) -> list[tuple[int, int, int]]:
    """(expert, row_lo, row_hi) — whole segments for average experts, HOT_CHUNK slices
    for the few hot ones (>2x avg, capped fanout) — the banked granularity co-design."""
    nonzero = [c for c in counts if c > 0]
    avg = max(1, sum(nonzero) // max(1, len(nonzero)))
    hot = set(sorted((e for e in range(E) if counts[e] > 2 * avg),
                     key=lambda e: -counts[e])[:HOT_FANOUT_CAP])
    segs, acc = [], 0
    for e in range(E):
        c, start = counts[e], acc
        acc += c
        step = HOT_CHUNK if e in hot else max(c, 1)
        while c > 0:
            take = min(c, step)
            segs.append((e, start, start + take))
            start += take
            c -= take
    return segs


def meta_tensors(segs, device):
    import torch

    pairs, ids = [], []
    for e, lo, hi in segs:
        if hi > lo:
            pairs += [lo, hi]
            ids.append(e)
    ids.append(-1)
    return (torch.tensor(pairs, dtype=torch.int32, device=device),
            torch.tensor(ids, dtype=torch.int32, device=device), len(ids))


def child_main(args) -> int:
    import torch
    import asym_gemm

    rank = args.rank
    plan = json.load(open(args.plan))
    fd = os.open(shm_path(args.tag), os.O_RDWR)
    mm = mmap.mmap(fd, ALIGN * (1 + len(plan["cases"])))
    os.close(fd)
    base = torch.frombuffer(mm, dtype=torch.uint8, count=mm.size())
    rc = torch.cuda.cudart().cudaHostRegister(base.data_ptr(), mm.size(), 0)
    assert int(rc) == 0, f"register rc={rc}"
    dev = torch.device("cuda", 0)

    gen = torch.Generator(device="cpu").manual_seed(1234)
    b_bank = (torch.randn(E, N, K, generator=gen, dtype=torch.bfloat16) * 0.02).pin_memory()
    results = []
    for ci, case in enumerate(plan["cases"]):
        counts = case["counts"]
        m_total = sum(counts)
        a_full = torch.randn(m_total, K, device=dev, dtype=torch.bfloat16)
        d_full = torch.empty(m_total, N, device=dev, dtype=torch.bfloat16)
        counters = base[ALIGN * (1 + ci): ALIGN * (1 + ci) + 12].view(torch.int32)

        # OWNED mode: fixed expert-ownership partition
        own_lo, own_hi = (0, E // 2) if rank == 0 else (E // 2, E)
        segs_all = chunk_segments(counts)
        segs_own = [s for s in segs_all if own_lo <= s[0] < own_hi]
        off_o, ids_o, ls_o = meta_tensors(segs_own, dev)
        # QUEUE mode: union list, shared counters
        off_q, ids_q, ls_q = meta_tensors(segs_all, dev)

        def run(mode: str) -> float:
            # CUDA-event BUSY time per rank (launch/sync floor excluded); barrier-aligned
            # starts; rep 0 = JIT warm, dropped.
            busys = []
            for rep in range(args.reps + 1):
                if rank == 0:
                    counters.zero_()
                barrier(args.tag, f"c{ci}_{mode}_{rep}a", rank, 2)
                torch.cuda.synchronize()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                if mode == "owned":
                    if ls_o > 1:
                        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                            a_full, b_bank, d_full, off_o, ids_o, ls_o, "nk", False)
                else:
                    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(
                        a_full, b_bank, d_full, off_q, ids_q, ls_q, counters, rank, "nk", False)
                ev1.record()
                torch.cuda.synchronize()
                busys.append(ev0.elapsed_time(ev1) / 1e3)
                barrier(args.tag, f"c{ci}_{mode}_{rep}b", rank, 2)
            return sum(busys[1:]) / max(1, len(busys) - 1)

        owned_s = run("owned")
        queue_s = run("queue")
        results.append({"case": case["name"], "alpha": case["alpha"],
                        "owned_s": owned_s, "queue_s": queue_s})
    json.dump(results, open(shm_path(args.tag) + f".res{rank}.json", "w"))
    return 0


def parent_main(args) -> int:
    tag = str(os.getpid())
    picks = load_layer_counts(args.hist, args.layers.split(","))
    alphas = [None if a == "natural" else float(a) for a in args.alphas.split(",")]
    cases = []
    for lname, counts in picks.items():
        for a in alphas:
            cases.append({
                "name": f"{lname}|alpha={'nat' if a is None else a}",
                "alpha": 0.0 if a is None else a,
                "counts": scale_counts(counts, args.m_total, 0.0 if a is None else a),
            })
    plan_path = shm_path(tag) + ".plan.json"
    size = ALIGN * (1 + len(cases))
    fd = os.open(shm_path(tag), os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size)
    mm[:] = b"\x00" * size
    os.close(fd)
    json.dump({"cases": cases}, open(plan_path, "w"))
    print(f"[parent] {len(cases)} cases; spawning children")
    procs = []
    for rank in range(2):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus.split(",")[rank])
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--role", "child", "--rank", str(rank),
             "--tag", tag, "--plan", plan_path, "--reps", str(args.reps)],
            env=env, cwd=REPO))
    rcs = [p.wait() for p in procs]
    out = {"cases": [], "child_rcs": rcs}
    try:
        r0 = json.load(open(shm_path(tag) + ".res0.json"))
        r1 = json.load(open(shm_path(tag) + ".res1.json"))
        for a, b in zip(r0, r1):
            owned_wall = max(a["owned_s"], b["owned_s"])
            owned_imb = abs(a["owned_s"] - b["owned_s"]) / max(a["owned_s"], b["owned_s"])
            queue_wall = max(a["queue_s"], b["queue_s"])
            queue_imb = abs(a["queue_s"] - b["queue_s"]) / max(a["queue_s"], b["queue_s"])
            out["cases"].append({
                "case": a["case"],
                "owned_wall_s": round(owned_wall, 4), "owned_imbalance": round(owned_imb, 4),
                "queue_wall_s": round(queue_wall, 4), "queue_imbalance": round(queue_imb, 4),
                "owned_over_queue": round(owned_wall / queue_wall, 3) if queue_wall else None,
            })
            print(out["cases"][-1])
    finally:
        for f in os.listdir("/dev/shm"):
            if f.startswith(os.path.basename(shm_path(tag))):
                try:
                    os.unlink("/dev/shm/" + f)
                except FileNotFoundError:
                    pass
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[parent] -> {args.out}")
    return 0 if all(rc == 0 for rc in rcs) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="parent", choices=["parent", "child"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--hist", default="")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--m-total", type=int, default=1280000)
    ap.add_argument("--alphas", default="natural,0.25,0.5,0.75")
    ap.add_argument("--layers", default="worst,median")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.role == "child":
        return child_main(args)
    assert args.hist, "--hist required (capture via ASYM_EP_STATS=1)"
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
