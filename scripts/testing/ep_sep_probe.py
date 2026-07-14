"""PR-5 (fix_gb200_ep.md S6): standalone validation of the true-sEP armed launch —
two processes, shm-registered pinned ctrl/X/D buffers (no fabric), union queue +
steal + gather, BITWISE check vs the plain contiguous kernel on each rank's own
segments. Case 2 forces cross-rank imbalance (rank0 3x rows) so stealing + gather
actually fire; 3 iterations exercise the ring + rotating flags.

  .venv/bin/python scripts/testing/ep_sep_probe.py [--gpus 2,3] [--mode queue|plan]

--mode plan validates the asym_sepplan2 flavor (count-computed cut, private
counter blocks, fabricated meet point) over the SAME cases and bitwise check.
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
sys.path.insert(0, REPO)

E, N, K = 128, 768, 2048
MAX_ROWS = 1 << 20  # 1M rows/slot


def shm(tag: str) -> str:
    return f"/dev/shm/asym_seprobe_{tag}"


def barrier(tag: str, name: str, rank: int) -> None:
    base = shm(tag) + f".bar_{name}"
    open(f"{base}.{rank}", "w").close()
    t0 = time.time()
    while any(not os.path.exists(f"{base}.{r}") for r in range(2)):
        if time.time() - t0 > 300:
            raise RuntimeError(f"barrier {name} timeout")
        time.sleep(0.005)


def child(args) -> int:
    import torch
    import asym_gemm
    from asym_gemm.training import ep_sep

    rank = args.rank
    torch.manual_seed(7 + rank)
    dev = torch.device("cuda", 0)

    ctrl_ints = ep_sep.ctrl_ints_needed()
    ctrl_bytes = (ctrl_ints * 4 + 4095) // 4096 * 4096
    slot_bytes = MAX_ROWS * K * 2
    dslot_bytes = MAX_ROWS * N * 2
    total = ctrl_bytes + 2 * ep_sep.RING * (slot_bytes + dslot_bytes)

    fd = os.open(shm(args.tag), os.O_RDWR)
    mm = mmap.mmap(fd, total)
    os.close(fd)
    base = torch.frombuffer(mm, dtype=torch.uint8, count=total)
    rc = torch.cuda.cudart().cudaHostRegister(base.data_ptr(), total, 0)
    assert int(rc) == 0, f"cudaHostRegister rc={rc}"

    ctrl = base[:ctrl_ints * 4].view(torch.int32)
    off = ctrl_bytes
    x_slots, d_slots = [[], []], [[], []]
    for r in range(2):
        for i in range(ep_sep.RING):
            x_slots[r].append(base[off:off + slot_bytes].view(torch.bfloat16))
            off += slot_bytes
    for r in range(2):
        for i in range(ep_sep.RING):
            d_slots[r].append(base[off:off + dslot_bytes].view(torch.bfloat16))
            off += dslot_bytes
    st = ep_sep.install_buffers(rank=rank, world=2, ctrl=ctrl, x_slots=x_slots, d_slots=d_slots)

    gen = torch.Generator(device="cpu").manual_seed(1234)
    b_bank = (torch.randn(E, N, K, generator=gen, dtype=torch.bfloat16) * 0.02).pin_memory()

    results = {}
    # case 1: balanced-ish small (streaming-bound); case 2: rank0 3x rows (steal fires);
    # case 3: repeat of case 1 (ring/flag reuse)
    # segment rows are BLOCK_M(128)-aligned — the e2e path guarantees this via
    # _pad_grouped_input_for_asym; UNALIGNED segments corrupt neighbors through
    # store-tile overhang across SIDES (the PR-5 receipt) and are out of contract.
    cases = [
        ("bal", 1536 if rank == 0 else 1408),      # rows/seg (x128 segs)
        ("skew", 3456 if rank == 0 else 1152),     # 3:1 => crossing fires
        ("decline", 4608 if rank == 0 else 1536),  # rank0 over floor => BOTH fall back
        ("bal2", 1536 if rank == 0 else 1408),
    ]
    for name, per in cases:
        m = per * E
        segs = []
        acc = 0
        for e in range(E):
            segs.append((e, acc, acc + per))
            acc += per
        a = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        d_ref = torch.empty(m, N, device=dev, dtype=torch.bfloat16)
        d_armed = torch.empty(m, N, device=dev, dtype=torch.bfloat16)

        pairs = torch.tensor([[lo, hi] for _, lo, hi in segs], dtype=torch.int32, device=dev).reshape(-1)
        ids = torch.tensor([e for e, _, _ in segs] + [-1], dtype=torch.int32, device=dev)
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a, b_bank, d_ref, pairs, ids, len(segs) + 1, "nk", False)
        torch.cuda.synchronize()

        barrier(args.tag, f"{name}_pre", rank)
        t0 = time.perf_counter()
        handled = st.try_armed(asym_gemm, a, b_bank, d_armed, segs, "nk", False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if name == "decline":
            assert not handled, "decline case must fall back on BOTH ranks"
            results[name] = {"bitwise": True, "fallback": True, "wall_ms": round(wall * 1e3, 2), "m": m}
        else:
            assert handled, f"armed launch declined unexpectedly ({name})"
            bitwise = torch.equal(d_armed, d_ref)
            max_diff = float((d_armed.float() - d_ref.float()).abs().max())
            results[name] = {"bitwise": bitwise, "max_diff": max_diff,
                             "wall_ms": round(wall * 1e3, 2), "m": m}
            if not bitwise:
                badmask = d_armed != d_ref
                bad = badmask.any(dim=1).nonzero().flatten()
                badcols = badmask.any(dim=0).nonzero().flatten()
                seq_used = st.seq - 1
                ctr = st._counters(seq_used)
                # column blocks (BLOCK_N from n_blk=12 over N=768 => 64)
                cb = sorted(set((badcols // 64).tolist()))
                per = m // E
                seg_of = [(int(r) // max(per, 1)) for r in bad[:2000]]
                segs_hit = sorted(set(seg_of))
                zeros = float((d_armed[bad] == 0).float().mean())
                in_seg = sorted(set(int(r) % max(per, 1) for r in bad[:200]))[:16]
                # split the world: is the PEER'S STAGE correct for the bad rows?
                ring_used = (st.seq - 1) % 4
                stage = st.d_slots[1 - rank][ring_used][: m * N].view(m, N)
                stage_rows = stage[bad.cpu()].cuda()
                stage_match_ref = float((stage_rows == d_ref[bad]).float().mean())
                stage_match_armed = float((stage_rows == d_armed[bad]).float().mean())
                print(f"[rank{rank}] {name} STAGE match_ref={stage_match_ref:.3f} "
                      f"match_armed={stage_match_armed:.3f}", flush=True)
                results[name]["debug"] = {
                    "bad_rows": int(bad.numel()),
                    "bad_lo": int(bad.min()), "bad_hi": int(bad.max()),
                    "bad_col_blocks64": cb,
                    "segs_hit": segs_hit[:8],
                    "frac_zero": round(zeros, 3),
                    "in_seg_offsets": in_seg,
                    "counters": [int(ctr[i]) for i in range(3)],
                }
                print(f"[rank{rank}] {name} BAD n={int(bad.numel())} segs={segs_hit[:8]} "
                      f"zeros={zeros:.2f} inseg={in_seg[:10]} colblk={cb[:4]}.. "
                      f"ctr={[int(ctr[i]) for i in range(3)]}", flush=True)
        barrier(args.tag, f"{name}_post", rank)

    results["stats"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.stats.items()}
    json.dump(results, open(shm(args.tag) + f".res{rank}.json", "w"))
    print(f"[rank{rank}] {results}", flush=True)
    return 0


def parent(args) -> int:
    tag = str(os.getpid())
    from asym_gemm.training import ep_sep  # for sizes only (no cuda in parent)

    ctrl_ints = ep_sep.ctrl_ints_needed()
    ctrl_bytes = (ctrl_ints * 4 + 4095) // 4096 * 4096
    total = ctrl_bytes + 2 * ep_sep.RING * (MAX_ROWS * K * 2 + MAX_ROWS * N * 2)
    fd = os.open(shm(tag), os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
    os.ftruncate(fd, total)
    os.close(fd)
    print(f"[parent] shm {total/1e9:.1f} GB, mode={args.mode}, spawning on GPUs {args.gpus}")
    procs = []
    for rank in range(2):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[rank]
        env["ASYM_EP_SEP"] = "1"
        env["ASYM_EP_SEP_MODE"] = args.mode
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--role", "child",
             "--rank", str(rank), "--tag", tag], env=env, cwd=REPO))
    rcs = [p.wait() for p in procs]
    ok = all(rc == 0 for rc in rcs)
    verdict = {}
    try:
        for r in range(2):
            verdict[f"rank{r}"] = json.load(open(shm(tag) + f".res{r}.json"))
        allbit = all(v["bitwise"] for r in verdict.values() for k, v in r.items() if k != "stats")
        print(f"PR5_{'PASS' if (ok and allbit) else 'FAIL'} mode={args.mode} bitwise={allbit}")
    finally:
        for f in os.listdir("/dev/shm"):
            if f.startswith(os.path.basename(shm(tag))):
                try:
                    os.unlink("/dev/shm/" + f)
                except FileNotFoundError:
                    pass
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="parent", choices=["parent", "child"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--mode", default="queue", choices=["queue", "plan"])
    args = ap.parse_args()
    return child(args) if args.role == "child" else parent(args)


if __name__ == "__main__":
    sys.exit(main())
