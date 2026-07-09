"""shared-fabric probes PR-1/2/3 (fix_gb200_ep.md S1; isolated kernel/system class).

PR-1  /dev/shm mmap + ONE cudaHostRegister per process: TIME it (the dp2 pain point was
      2x cudaHostAlloc ~10+ min; register of resident tmpfs pages should be far faster).
PR-2  BOTH GPUs, TWO PROCESSES, stream B from the SAME registered fabric range
      concurrently via the asym GEMM -> GB/s per lane (banked in-process number: 174.7).
PR-3  cross-process union queue: the ep_queued kernel's atomicAdd_system counters live on
      a fabric page shared by both processes; side 0 pops front, side 1 pops back.
      GATES: d0 + d1 == static reference BITWISE (disjoint claims), head+tail == n_items.

Usage:  .venv/bin/python scripts/testing/shared_fabric_probe.py [--gb 8] [--out probe.json]
Parent spawns one child per GPU (CUDA_VISIBLE_DEVICES narrowed); file barriers sync phases.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALIGN = 4096

# fabric layout (parent writes; offsets in bytes)
#   [0]                header page: int64 sizes
#   [4096]             counters page: int32[3] queue counters (PR-3)
#   [8192]             PR-3 bank: bf16 [G3, N3, K3]
#   [bank3_end aligned] PR-2 bank: bf16 [N2, K2] repeated logically (one big 2D bank)
G3, N3, K3, ROWS3 = 32, 256, 512, 512  # PR-3 grouped geometry (M = G3*ROWS3)
N2, K2 = 131072, 2048                   # PR-2 bandwidth bank: 512 MiB bf16


def _align(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


def fabric_path(tag: str) -> str:
    return f"/dev/shm/asym_fabric_probe_{tag}"


def layout(gb: float) -> dict:
    bank3 = G3 * N3 * K3 * 2
    bank2 = N2 * K2 * 2
    off_counters = ALIGN
    off_bank3 = 2 * ALIGN
    off_bank2 = _align(off_bank3 + bank3)
    pad_target = int(gb * (1 << 30))
    total = max(_align(off_bank2 + bank2), _align(pad_target))
    return {
        "off_counters": off_counters,
        "off_bank3": off_bank3,
        "off_bank2": off_bank2,
        "bank3_bytes": bank3,
        "bank2_bytes": bank2,
        "total": total,
    }


def barrier(tag: str, name: str, rank: int, world: int, timeout: float = 600.0) -> None:
    base = fabric_path(tag) + f".bar_{name}"
    open(f"{base}.{rank}", "w").close()
    deadline = time.time() + timeout
    while any(not os.path.exists(f"{base}.{r}") for r in range(world)):
        if time.time() > deadline:
            raise RuntimeError(f"barrier {name} timed out (rank {rank})")
        time.sleep(0.02)


def child_main(args: argparse.Namespace) -> int:
    import torch

    lay = layout(args.gb)
    rank = args.rank
    res: dict = {"rank": rank}
    fd = os.open(fabric_path(args.tag), os.O_RDWR)
    mm = mmap.mmap(fd, lay["total"])
    os.close(fd)
    base = torch.frombuffer(mm, dtype=torch.uint8, count=lay["total"])

    dev = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES narrowed by the parent
    torch.cuda.init()

    # ---- PR-1: register the whole mapping once ----
    t0 = time.perf_counter()
    rc = torch.cuda.cudart().cudaHostRegister(base.data_ptr(), lay["total"], 0)
    res["pr1_register_seconds"] = time.perf_counter() - t0
    res["pr1_register_rc"] = int(rc)
    if int(rc) != 0:
        res["fail"] = f"cudaHostRegister rc={int(rc)}"
        return _emit(args, res)
    probe_view = base[lay["off_bank2"] : lay["off_bank2"] + lay["bank2_bytes"]]
    res["pr1_is_pinned"] = bool(probe_view.is_pinned())

    from asym_gemm.training.frozen_linear import _asym_bf16_nt

    b2 = base[lay["off_bank2"] : lay["off_bank2"] + lay["bank2_bytes"]].view(torch.bfloat16).view(N2, K2)
    a2 = torch.randn(256, K2, device=dev, dtype=torch.bfloat16)
    # JIT warm + correctness sample
    out = _asym_bf16_nt(a2, b2[:4096])
    torch.cuda.synchronize()
    res["pr2_warm_ok"] = bool(out.shape == (256, 4096))

    # ---- PR-2: concurrent streaming bandwidth (both ranks hammer the SAME bank) ----
    reps = args.reps
    barrier(args.tag, "pr2", rank, args.world)
    t0 = time.perf_counter()
    for _ in range(reps):
        _asym_bf16_nt(a2, b2)  # streams the full 512 MiB bank from host per call
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    res["pr2_gbps_per_lane"] = lay["bank2_bytes"] * reps / dt / 1e9
    res["pr2_seconds"] = dt

    _emit(args, res)  # partial emit: PR-1/2 survive a PR-3 failure

    # ---- PR-3: cross-process union queue ----
    import asym_gemm

    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued"):
        res["pr3_skip"] = "_C lacks ep_queued entry (rebuild: setup.py build_ext --inplace)"
        return _emit(args, res)

    counters = base[lay["off_counters"] : lay["off_counters"] + 12].view(torch.int32)
    b3 = base[lay["off_bank3"] : lay["off_bank3"] + lay["bank3_bytes"]].view(torch.bfloat16).view(G3, N3, K3)
    m_total = G3 * ROWS3
    gen = torch.Generator(device="cpu").manual_seed(1234)  # IDENTICAL a on both ranks
    a3 = torch.randn(m_total, K3, generator=gen, dtype=torch.bfloat16).to(dev)
    d3 = torch.zeros(m_total, N3, device=dev, dtype=torch.bfloat16)
    pairs, ids = [], []
    for e in range(G3):
        pairs += [e * ROWS3, (e + 1) * ROWS3]
        ids.append(e)
    ids.append(-1)
    offsets = torch.tensor(pairs, dtype=torch.int32, device=dev)
    experts = torch.tensor(ids, dtype=torch.int32, device=dev)
    list_size = len(ids)
    if rank == 0:
        counters.zero_()
    barrier(args.tag, "pr3a", rank, args.world)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(
        a3, b3, d3, offsets, experts, list_size, counters, rank, "nk", False
    )
    torch.cuda.synchronize()
    barrier(args.tag, "pr3b", rank, args.world)
    res["pr3_counters"] = [int(x) for x in counters.tolist()]
    res["pr3_rows_claimed"] = int((d3.abs().sum(dim=1) > 0).sum().item())
    torch.save(d3.cpu(), fabric_path(args.tag) + f".d{rank}.pt")
    if rank == 0:
        ref = torch.zeros_like(d3)
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a3, b3, ref, offsets, experts, list_size, "nk", False
        )
        torch.cuda.synchronize()
        torch.save(ref.cpu(), fabric_path(args.tag) + ".ref.pt")
    barrier(args.tag, "pr3c", rank, args.world)
    if rank == 0:
        d0 = torch.load(fabric_path(args.tag) + ".d0.pt", weights_only=True)
        d1 = torch.load(fabric_path(args.tag) + ".d1.pt", weights_only=True)
        ref = torch.load(fabric_path(args.tag) + ".ref.pt", weights_only=True)
        res["pr3_union_bitwise"] = bool(torch.equal(d0 + d1, ref))
        overlap = ((d0.abs().sum(1) > 0) & (d1.abs().sum(1) > 0)).sum().item()
        res["pr3_claim_overlap_rows"] = int(overlap)

    # ---- PR-4: STEAL — union list over TWO different packs, forced skew ----
    _emit(args, res)
    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal"):
        res["pr4_skip"] = "_C lacks ep_steal entry"
        return _emit(args, res)
    # geometry: rank0 owns 24 segments, rank1 owns 8 (rank1 drains its back section
    # early and steals rank0's items). Packs differ per rank (different X!).
    segs = (24, 8)
    rows_per = ROWS3
    m_local = segs[rank] * rows_per
    m_peer_rows = segs[1 - rank] * rows_per
    gen0 = torch.Generator(device="cpu").manual_seed(500)
    gen1 = torch.Generator(device="cpu").manual_seed(501)
    x_full = {
        0: torch.randn(segs[0] * rows_per, K3, generator=gen0, dtype=torch.bfloat16),
        1: torch.randn(segs[1] * rows_per, K3, generator=gen1, dtype=torch.bfloat16),
    }
    # fabric layout for PR-4 (carved after the PR-2 bank, parent pre-sized generously):
    lay4_x0 = _align(lay["off_bank2"] + lay["bank2_bytes"])
    lay4_x1 = _align(lay4_x0 + x_full[0].numel() * 2)
    lay4_d0 = _align(lay4_x1 + x_full[1].numel() * 2)          # staging WRITTEN BY rank0
    lay4_d1 = _align(lay4_d0 + segs[1] * rows_per * N3 * 2)    # staging WRITTEN BY rank1
    lay4_cnt = _align(lay4_d1 + segs[0] * rows_per * N3 * 2)
    need = lay4_cnt + ALIGN
    if need > lay["total"]:
        res["pr4_skip"] = f"fabric too small for PR-4 ({need} > {lay['total']}); rerun with --gb >= 10"
        return _emit(args, res)
    xf = {
        0: base[lay4_x0 : lay4_x0 + x_full[0].numel() * 2].view(torch.bfloat16).view(-1, K3),
        1: base[lay4_x1 : lay4_x1 + x_full[1].numel() * 2].view(torch.bfloat16).view(-1, K3),
    }
    # d_peer written by rank i holds rows in the PEER pack's coords:
    stag = {
        0: base[lay4_d0 : lay4_d0 + segs[1] * rows_per * N3 * 2].view(torch.bfloat16).view(-1, N3),
        1: base[lay4_d1 : lay4_d1 + segs[0] * rows_per * N3 * 2].view(torch.bfloat16).view(-1, N3),
    }
    cnt4 = base[lay4_cnt : lay4_cnt + 12].view(torch.int32)
    if rank == 0:
        xf[0].copy_(x_full[0])
        xf[1].copy_(x_full[1])
        stag[0].zero_()
        stag[1].zero_()
        cnt4.zero_()
    barrier(args.tag, "pr4a", rank, args.world)
    a_local = x_full[rank].to(dev)
    d_local = torch.zeros(m_local, N3, device=dev, dtype=torch.bfloat16)  # holes = stolen
    # union metadata: [rank0 segs (rank0 coords) | rank1 segs (rank1 coords)]
    pairs, ids = [], []
    for r in (0, 1):
        for s in range(segs[r]):
            pairs += [s * rows_per, (s + 1) * rows_per]
            ids.append((s * 7 + r) % G3)  # spread experts
    ids.append(-1)
    offsets4 = torch.tensor(pairs, dtype=torch.int32, device=dev)
    experts4 = torch.tensor(ids, dtype=torch.int32, device=dev)
    list4 = len(ids)
    n_own_boundary = segs[0]  # boundary in SEGMENT units (start of rank1's section)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal(
        a_local, b3, d_local, xf[1 - rank], stag[rank],
        offsets4, experts4, list4, cnt4, rank, n_own_boundary, "nk", False
    )
    torch.cuda.synchronize()
    barrier(args.tag, "pr4b", rank, args.world)
    head, tail = int(cnt4[1]), int(cnt4[2])
    res["pr4_counters"] = [int(cnt4[0]), head, tail]
    total_items = None  # items = segments * n_blk (JIT-internal); work in SEGMENT units
    # segment-claim ranges (contiguous by construction): rank0 claimed segments
    # [0, head/n_blk), rank1 claimed [n_segs - tail/n_blk, n_segs). Derive n_blk from
    # head+tail == n_segs * n_blk (all items claimed exactly once).
    n_segs = segs[0] + segs[1]
    if (head + tail) % n_segs == 0:
        n_blk = (head + tail) // n_segs
    else:
        n_blk = 0
    res["pr4_n_blk"] = n_blk
    if n_blk:
        total_items = n_segs * n_blk
        n_own_items = n_own_boundary * n_blk
        res["pr4_items"] = {"total": total_items, "head": head, "tail": tail,
                            "complete": head + tail == total_items}
        # STEAL GATHER at ITEM-TILE granularity (the meeting point can split a segment
        # mid-way: local and stolen tiles of one segment differ by COLUMN strip):
        # side0 claimed items [0, head); side1 claimed [total-tail, total). Items of
        # rank0's section stolen by rank1 = [head, n_own_items) (when head < n_own_items).
        assert N3 % n_blk == 0, "probe geometry: N must divide by n_blk"
        bn = N3 // n_blk
        if rank == 0 and head < n_own_items:
            stag_dev = stag[1].to(dev)  # staging holds rank0-coord rows written by rank1
            gathered = 0
            for item in range(head, n_own_items):
                seg, cb = divmod(item, n_blk)
                rows = slice(seg * rows_per, (seg + 1) * rows_per)
                cols = slice(cb * bn, (cb + 1) * bn)
                d_local[rows, cols] = stag_dev[rows, cols]
                gathered += 1
            res["pr4_gathered_item_tiles"] = gathered
        if rank == 1:
            res["pr4_stole_items"] = max(0, tail - (total_items - n_own_items))
    # reference: full local pack through the plain grouped kernel (local metadata only)
    pairs_l, ids_l = [], []
    for s in range(segs[rank]):
        pairs_l += [s * rows_per, (s + 1) * rows_per]
        ids_l.append(ids[s + (0 if rank == 0 else segs[0])])
    ids_l.append(-1)
    ref_l = torch.zeros_like(d_local)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a_local, b3, ref_l,
        torch.tensor(pairs_l, dtype=torch.int32, device=dev),
        torch.tensor(ids_l, dtype=torch.int32, device=dev),
        len(ids_l), "nk", False
    )
    torch.cuda.synchronize()
    res["pr4_local_bitwise"] = bool(torch.equal(d_local, ref_l))
    return _emit(args, res)


def _emit(args: argparse.Namespace, res: dict) -> int:
    with open(fabric_path(args.tag) + f".res{args.rank}.json", "w") as fh:
        json.dump(res, fh)
    return 0


def parent_main(args: argparse.Namespace) -> int:
    tag = args.tag or str(os.getpid())
    lay = layout(args.gb)
    path = fabric_path(tag)
    for f in (path,):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
    os.ftruncate(fd, lay["total"])
    mm = mmap.mmap(fd, lay["total"])
    os.close(fd)
    print(f"[parent] fabric {path} total={lay['total']/(1<<30):.1f} GiB; writing banks...")
    import numpy as np

    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    b3 = (rng.standard_normal(G3 * N3 * K3) * 0.05).astype(np.float32).astype(np.float16)
    mm[lay["off_bank3"] : lay["off_bank3"] + lay["bank3_bytes"]] = _f16_to_bf16_bytes(b3)
    chunk = 1 << 24
    total2 = lay["bank2_bytes"]
    fill = _f16_to_bf16_bytes((rng.standard_normal(chunk // 2) * 0.05).astype(np.float32).astype(np.float16))
    off = lay["off_bank2"]
    while total2 > 0:
        take = min(chunk, total2)
        mm[off : off + take] = fill[:take]
        off += take
        total2 -= take
    # touch ONLY the padding tail (banks already written) so register pins RESIDENT pages,
    # matching production where every registered page holds bank bytes
    step = 1 << 20
    tail_start = _align(lay["off_bank2"] + lay["bank2_bytes"])
    for off in range(tail_start, lay["total"], step):
        mm[off : off + 1] = b"\x00"
    print(f"[parent] banks written in {time.perf_counter()-t0:.1f}s; spawning children")
    procs = []
    for rank in range(args.world):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus.split(",")[rank])
        cmd = [sys.executable, os.path.abspath(__file__), "--role", "child", "--rank", str(rank),
               "--world", str(args.world), "--tag", tag, "--gb", str(args.gb), "--reps", str(args.reps)]
        procs.append(subprocess.Popen(cmd, env=env, cwd=REPO))
    rcs = [p.wait() for p in procs]
    out: dict = {"layout_gib": lay["total"] / (1 << 30), "child_rcs": rcs}
    for rank in range(args.world):
        try:
            out[f"rank{rank}"] = json.load(open(path + f".res{rank}.json"))
        except FileNotFoundError:
            out[f"rank{rank}"] = {"fail": "no result file"}
    r0, r1 = out.get("rank0", {}), out.get("rank1", {})
    out["verdicts"] = {
        "PR1_register_seconds": [r0.get("pr1_register_seconds"), r1.get("pr1_register_seconds")],
        "PR1_pinned": bool(r0.get("pr1_is_pinned")) and bool(r1.get("pr1_is_pinned")),
        "PR2_gbps_per_lane": [r0.get("pr2_gbps_per_lane"), r1.get("pr2_gbps_per_lane")],
        "PR3_union_bitwise": r0.get("pr3_union_bitwise"),
        "PR3_counters": r0.get("pr3_counters"),
        "PR3_rows_claimed": [r0.get("pr3_rows_claimed"), r1.get("pr3_rows_claimed")],
        "PR3_overlap_rows": r0.get("pr3_claim_overlap_rows"),
    }
    print(json.dumps(out["verdicts"], indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[parent] full results -> {args.out}")
    if not args.keep:
        for suffix in ("", ".d0.pt", ".d1.pt", ".ref.pt", ".res0.json", ".res1.json"):
            try:
                os.unlink(path + suffix)
            except FileNotFoundError:
                pass
        for f in os.listdir("/dev/shm"):
            if f.startswith(os.path.basename(path) + ".bar_"):
                try:
                    os.unlink("/dev/shm/" + f)
                except FileNotFoundError:
                    pass
    pr3_skip = r0.get("pr3_skip") or r1.get("pr3_skip")
    if pr3_skip:
        out["verdicts"]["PR3_skip"] = pr3_skip
    pr12_ok = bool(out["verdicts"]["PR1_pinned"]) and all(
        isinstance(x, float) for x in out["verdicts"]["PR2_gbps_per_lane"]
    )
    pr3_ok = bool(pr3_skip) or (
        bool(out["verdicts"]["PR3_union_bitwise"]) and out["verdicts"]["PR3_overlap_rows"] == 0
    )
    ok = pr12_ok and pr3_ok
    print("PROBE", ("PASS (PR3 SKIPPED)" if pr3_skip else "PASS") if ok else "FAIL")
    return 0 if ok else 1


def _f16_to_bf16_bytes(f16: "np.ndarray") -> bytes:
    import numpy as np

    f32 = f16.astype(np.float32)
    u32 = f32.view(np.uint32)
    bf16 = (u32 >> 16).astype(np.uint16)
    return bf16.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="parent", choices=["parent", "child"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gb", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--out", default="")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    if args.role == "child":
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
