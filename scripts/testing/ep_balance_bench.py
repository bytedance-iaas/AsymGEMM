"""S5a/S6 BALANCING + STREAMING microbench (fix_gb200_ep.md): assignment policies
over REAL recorded routing, two processes, real bank shapes, tuned-alpha injection.

MODES (the policy A/B/C/D; transport identical — every rank holds the union A
locally and B streams from the same pinned host bank — so walls isolate the
ASSIGNMENT POLICY + its bank-streaming footprint):
  owned : rank r executes exactly experts [r*E/2,(r+1)*E/2)'s segments (fixed
          ownership — the classic-EP disease rung).
  sdp   : rank r executes its OWN HALF of every expert's rows over ALL experts
          (production shared-bank streaming DP shape).
  sep   : chunked-LPT planner over the UNION counts assigns whole experts (mega
          experts chunked) to ranks ~evenly => bank-once consolidation (S6 true-sEP
          planner flavor).
  queue : union list, SHARED system-scope counters, side 0 pops front / 1 back
          (S6 emergent flavor; hot-expert chunking at EP_HOT_CHUNK_ROWS).
Wall per mode = max(rank CUDA-event busy); imbalance = |b0-b1|/max. B-bytes are
ANALYTIC: executed segments x gate-bank bytes (N*K*2B); chunked segments re-stream
the bank once per chunk; queue per-side split read from the counters when they
reconcile (head+tail==claimed), else NA.

Usage: scripts/testing/ep_balance_bench.sh (wrapper) or:
  .venv/bin/python scripts/testing/ep_balance_bench.py \
      --hist profiling_gb200ep_sg/ep_hist_q3_s20000.json \
      [--modes owned,sdp,sep,queue] [--m-total 5120000] [--alphas natural,0.15]
      [--layers worst,median] [--reps 3] [--gpus 2,3] [--out bench.json]
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
BANK_BYTES = N * K * 2            # one expert's gate bank, bf16
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


def scale_counts(counts: list[int], m_total: int, alpha: float,
                 gamma: float = 1.0) -> list[int]:
    total = sum(counts) or 1
    if gamma != 1.0:
        # tier-2 severity knob (literature-standard Zipf-shaping): sharpen the REAL
        # recorded distribution p_i^gamma / sum — preserves its gradual multi-expert
        # character while dialing concentration (gamma=1 natural; ->inf approaches
        # the single-hot-expert bound).
        sharp = [(c / total) ** gamma for c in counts]
        z = sum(sharp) or 1.0
        scaled = [max(0, round(s / z * m_total)) for s in sharp]
    else:
        scaled = [max(0, round(c * m_total / total)) for c in counts]
    if alpha > 0:  # tier-3 adversarial bound: alpha of ALL rows onto expert 0
        moved = 0
        for e in range(1, E):
            take = round(scaled[e] * alpha)
            scaled[e] -= take
            moved += take
        scaled[0] += moved
    hot = max(range(E), key=lambda e: scaled[e])
    scaled[hot] += m_total - sum(scaled)
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


def sdp_segments(counts: list[int], rank: int) -> list[tuple[int, int, int]]:
    """Rank's own half of every expert's rows (contiguous sub-range per segment)."""
    segs, acc = [], 0
    for e in range(E):
        c, start = counts[e], acc
        acc += c
        half = c // 2
        lo, hi = (start, start + half) if rank == 0 else (start + half, start + c)
        if hi > lo:
            segs.append((e, lo, hi))
    return segs


def sep_planner_segments(counts: list[int], rank: int) -> list[tuple[int, int, int]]:
    """Chunked-LPT over the union: whole experts to the lighter bin (locality is moot
    here — union A is rank-local); a mega-expert (> total/2) is split at the excess."""
    total = sum(counts)
    target = total / 2.0
    # expert -> (rows, row_lo) in the expert-major union layout
    starts, acc = [], 0
    for e in range(E):
        starts.append(acc)
        acc += counts[e]
    items: list[tuple[int, int, int]] = []  # (rows, expert, row_lo) whole or split
    for e in range(E):
        c = counts[e]
        if c <= 0:
            continue
        if c > target:  # mega-expert: keep target rows whole, spill the excess chunked
            items.append((int(target), e, starts[e]))
            spill_lo = starts[e] + int(target)
            spill = c - int(target)
            while spill > 0:
                take = min(spill, HOT_CHUNK)
                items.append((take, e, spill_lo))
                spill_lo += take
                spill -= take
        else:
            items.append((c, e, starts[e]))
    loads = [0, 0]
    mine: list[tuple[int, int, int]] = []
    for rows, e, lo in sorted(items, reverse=True):
        b = 0 if loads[0] <= loads[1] else 1
        loads[b] += rows
        if b == rank:
            mine.append((e, lo, lo + rows))
    mine.sort(key=lambda s: s[1])  # ascending offsets for the contiguous kernel
    return mine


def analytic_b_bytes(segs: list[tuple[int, int, int]]) -> tuple[int, int]:
    """(streamed B bytes, distinct experts): one full gate bank per EXECUTED SEGMENT
    (chunks re-stream), which is how the contiguous kernel consumes B."""
    return len([s for s in segs if s[2] > s[1]]) * BANK_BYTES, len({s[0] for s in segs})


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
    modes = args.modes.split(",")
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

        own_lo, own_hi = (0, E // 2) if rank == 0 else (E // 2, E)
        segs_all = chunk_segments(counts)
        seg_sets = {
            "owned": [s for s in segs_all if own_lo <= s[0] < own_hi],
            "sdp": sdp_segments(counts, rank),
            "sep": sep_planner_segments(counts, rank),
            "queue": segs_all,
        }

        def run(mode: str) -> dict:
            segs = seg_sets[mode]
            off, ids, ls = meta_tensors(segs, dev)
            busys = []
            for rep in range(args.reps + 1):  # rep 0 = JIT warm, dropped
                if rank == 0:
                    counters.zero_()
                barrier(args.tag, f"c{ci}_{mode}_{rep}a", rank, 2)
                torch.cuda.synchronize()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                if mode == "queue":
                    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(
                        a_full, b_bank, d_full, off, ids, ls, counters, rank, "nk", False)
                elif ls > 1:
                    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                        a_full, b_bank, d_full, off, ids, ls, "nk", False)
                ev1.record()
                torch.cuda.synchronize()
                busys.append(ev0.elapsed_time(ev1) / 1e3)
                barrier(args.tag, f"c{ci}_{mode}_{rep}b", rank, 2)
            out: dict = {"busy_s": sum(busys[1:]) / max(1, len(busys) - 1)}
            if mode == "queue":
                # per-side split from the shared counters of the LAST rep
                claimed, head, tail = (int(counters[i]) for i in range(3))
                n_items = len([s for s in segs if s[2] > s[1]])
                if claimed > 0 and head + tail == claimed:
                    share = (head if rank == 0 else tail) / claimed
                    total_b = n_items * BANK_BYTES  # item = (segment, n_block) slice
                    out["b_bytes"] = int(total_b * share)
                    out["experts"] = -1  # interleaved; per-side expert set unknown
                else:
                    out["b_bytes"] = -1
                    out["experts"] = -1
            else:
                b, ex = analytic_b_bytes(segs)
                out["b_bytes"] = b
                out["experts"] = ex
            out["rows"] = sum(hi - lo for _, lo, hi in segs)
            return out

        rec = {"case": case["name"], "alpha": case["alpha"], "m_total": m_total}
        for mode in modes:
            rec[mode] = run(mode)
        results.append(rec)
    json.dump(results, open(shm_path(args.tag) + f".res{rank}.json", "w"))
    return 0


def parent_main(args) -> int:
    tag = str(os.getpid())
    picks = load_layer_counts(args.hist, args.layers.split(","))
    modes = args.modes.split(",")
    cases = []
    for lname, counts in picks.items():
        for tok in args.alphas.split(","):
            tok = tok.strip()
            if tok == "natural":
                a, g, tag = 0.0, 1.0, "nat"
            elif tok.startswith("g"):   # tier-2 gamma token, e.g. g1.5
                a, g, tag = 0.0, float(tok[1:]), tok
            else:                        # tier-3 alpha token
                a, g, tag = float(tok), 1.0, f"a{tok}"
            cases.append({
                "name": f"{lname}|{tag}",
                "alpha": a,
                "counts": scale_counts(counts, args.m_total, a, gamma=g),
            })
    plan_path = shm_path(tag) + ".plan.json"
    size = ALIGN * (1 + len(cases))
    fd = os.open(shm_path(tag), os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size)
    mm[:] = b"\x00" * size
    os.close(fd)
    json.dump({"cases": cases}, open(plan_path, "w"))
    print(f"[parent] {len(cases)} cases x modes {modes} (m_total={args.m_total}); spawning")
    procs = []
    for rank in range(2):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus.split(",")[rank])
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--role", "child", "--rank", str(rank),
             "--tag", tag, "--plan", plan_path, "--reps", str(args.reps),
             "--modes", args.modes],
            env=env, cwd=REPO))
    rcs = [p.wait() for p in procs]
    out = {"m_total": args.m_total, "modes": modes, "cases": [], "child_rcs": rcs}
    try:
        r0 = json.load(open(shm_path(tag) + ".res0.json"))
        r1 = json.load(open(shm_path(tag) + ".res1.json"))
        for a, b in zip(r0, r1):
            row = {"case": a["case"], "m_per_expert": a["m_total"] // E}
            for m in modes:
                wall = max(a[m]["busy_s"], b[m]["busy_s"])
                imb = abs(a[m]["busy_s"] - b[m]["busy_s"]) / max(wall, 1e-9)
                row[m] = {
                    "wall_s": round(wall, 5), "imbalance": round(imb, 4),
                    "b_mb": [round(a[m]["b_bytes"] / 1e6, 1), round(b[m]["b_bytes"] / 1e6, 1)],
                    "experts": [a[m]["experts"], b[m]["experts"]],
                    "rows": [a[m]["rows"], b[m]["rows"]],
                }
            out["cases"].append(row)
            compact = {m: (row[m]["wall_s"], row[m]["imbalance"], row[m]["b_mb"]) for m in modes}
            print(f"  {row['case']}: " + " | ".join(
                f"{m} w={v[0]*1e3:.2f}ms imb={v[1]:.3f} B={v[2]}MB" for m, v in compact.items()))
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
    ap.add_argument("--m-total", type=int, default=5120000)
    ap.add_argument("--alphas", default="natural,0.25,0.5,0.75")
    ap.add_argument("--layers", default="worst,median")
    ap.add_argument("--modes", default="owned,sdp,sep,queue")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.role == "child":
        return child_main(args)
    assert args.hist, "--hist required (capture via ASYM_EP_STATS=1)"
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
