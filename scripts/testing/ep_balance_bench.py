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

SCOPE (--scope / env SCOPE) widens the timed pipeline; same modes/cases:
  gemm    : one gate/up-shaped grouped GEMM (Table 1, unchanged).
  experts : gate GEMM + up GEMM + SiLU*mul + down GEMM (all three banks streamed;
            B-bytes = 3 banks per executed segment; elementwise on executed rows).
  moe     : + router (own half of tokens, identical on both ranks) + token gather
            into the packed layout (charged per EXECUTED rows — the owner-side
            receive volume is what skews) + weighted combine. Combine is charged at
            m/2 rows per rank for EVERY mode (real systems combine AFTER token
            return, i.e. balanced) and modeled as scale+scatter-write, NOT
            torch bf16 index_add (whose 2-byte CAS atomics cost 53 ms/2.56M rows —
            a torch artifact that would drown the story; accumulation structure is
            identical across modes anyway). Dispatch INDEX CONSTRUCTION off the
            clock (same O(m) sort in every system); NO cross-GPU token movement
            (transport-identical design, favors EP).
  queue-mode gather/act are charged at m/2 rows per rank (the queue splits work
  near-evenly; GEMM stages run as three queued races).

Usage: scripts/testing/ep_balance_bench.sh (wrapper) or:
  .venv/bin/python scripts/testing/ep_balance_bench.py \
      --hist profiling_both_epstats/ep_hist_q3_s20000.json \
      [--modes owned,sdp,plan,queue] [--m-total 5120000] [--alphas natural,0.15]
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
# Default geometry: q3-30b gate/up (real shapes). Overridable per model via
# --geom E,N,K --topk --fused-gateup --shared-n (values VERIFIED from the HF
# configs; see MODEL presets in ep_balance_bench.sh).
E, N, K = 128, 768, 2048
TOPK = 8                          # routing fanout (n_tokens = m_total / TOPK)
FUSED_GATEUP = False              # llama4: ONE (2N, K) gate_up GEMM instead of two
SHARED_N = 0                      # shared-expert width (0 = none); moe scope only
BANK_BYTES = N * K * 2            # one expert's gate (or down) bank, bf16
HOT_CHUNK = int(os.environ.get("EP_HOT_CHUNK_ROWS", "8192"))
HOT_FANOUT_CAP = 8


def set_geometry(geom: str, topk: int, fused: bool, shared_n: int) -> None:
    global E, N, K, TOPK, FUSED_GATEUP, SHARED_N, BANK_BYTES
    E, N, K = (int(x) for x in geom.split(","))
    TOPK, FUSED_GATEUP, SHARED_N = topk, fused, shared_n
    BANK_BYTES = N * K * 2


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


def chunk_segments(counts: list[int], n_blk_min: int = 1,
                   n_floor: int = 0) -> list[tuple[int, int, int]]:
    """(expert, row_lo, row_hi) — whole segments for average experts, sliced hot ones
    (>2x avg, capped fanout) — the QUEUE's sharing granularity. FINE grain is what
    makes stealing bulletproof (coarse grains left 18-35% unshared when the mega
    expert parked at one list end — receipts in fix_gb200_ep_v2 RUN LOG); the only
    counter-pressure is bank re-streaming, bounded by the n_floor=4N-rows floor
    (re-stream <= ~25% of a chunk's A traffic). q3 shapes reduce exactly to the
    banked HOT_CHUNK=8192. n_blk_min is accepted for signature stability (the
    unit-count-targeting variants it enabled traded balance for wall — rejected)."""
    nonzero = [c for c in counts if c > 0]
    avg = max(1, sum(nonzero) // max(1, len(nonzero)))
    hot = set(sorted((e for e in range(E) if counts[e] > 2 * avg),
                     key=lambda e: -counts[e])[:HOT_FANOUT_CAP])
    segs, acc = [], 0
    for e in range(E):
        c, start = counts[e], acc
        acc += c
        if e in hot:
            step = ((max(n_floor, HOT_CHUNK) + 127) // 128) * 128
        else:
            step = max(c, 1)
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


def zipf_counts(s: float, m_total: int, seed: int) -> list[int]:
    """C1 (fix_gb200_ep_v2): literature-standard synthetic loads — share of the
    i-th busiest expert ~ 1/i^s, expert IDs assigned by a seeded permutation."""
    import random

    shares = [1.0 / ((r + 1) ** s) for r in range(E)]
    z = sum(shares)
    ids = list(range(E))
    random.Random(seed).shuffle(ids)
    counts = [0] * E
    for r, eid in enumerate(ids):
        counts[eid] = max(0, round(shares[r] / z * m_total))
    hot = max(range(E), key=lambda e: counts[e])
    counts[hot] += m_total - sum(counts)
    return counts


def owned_smart_segments(counts: list[int], rank: int) -> list[tuple[int, int, int]]:
    """C2 (fix_gb200_ep_v2): EP with the BEST possible placement — whole experts
    assigned to GPUs by LPT bin-packing (placement cannot split an expert)."""
    starts, acc = [], 0
    for e in range(E):
        starts.append(acc)
        acc += counts[e]
    loads = [0, 0]
    mine = []
    for c, e in sorted(((counts[e], e) for e in range(E)), reverse=True):
        if c <= 0:
            continue
        b = 0 if loads[0] <= loads[1] else 1
        loads[b] += c
        if b == rank:
            mine.append((e, starts[e], starts[e] + c))
    mine.sort(key=lambda x: x[1])
    return mine


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


def merged_runs(segs: list[tuple[int, int, int]]) -> list[list[int]]:
    """Maximal contiguous [lo, hi) row runs — keeps elementwise/gather/combine at a
    few big slices instead of one launch per (possibly chunked) segment."""
    runs: list[list[int]] = []
    for _, lo, hi in sorted(segs, key=lambda s: s[1]):
        if hi <= lo:
            continue
        if runs and runs[-1][1] == lo:
            runs[-1][1] = hi
        else:
            runs.append([lo, hi])
    return runs


def child_main(args) -> int:
    import torch
    import torch.nn.functional as F
    import asym_gemm

    rank = args.rank
    scope = args.scope
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

    def mk_bank(n_out: int, k_in: int):
        return (torch.randn(E, n_out, k_in, generator=gen, dtype=torch.bfloat16) * 0.02).pin_memory()

    # GEMM stage plan: (name, bank, k_in, n_out). scope=gemm runs stage 0 only.
    if FUSED_GATEUP:
        gemm_stages = [("gate_up", mk_bank(2 * N, K), K, 2 * N)]
    else:
        gemm_stages = [("gate", mk_bank(N, K), K, N)]
    if scope in ("experts", "moe", "layer"):
        if not FUSED_GATEUP:
            gemm_stages.append(("up", mk_bank(N, K), K, N))
        gemm_stages.append(("down", mk_bank(K, N), N, K))
    ws_gu = ws_down = None
    # scope=layer: WHOLE transformer layer = attention prelude + MoE block.
    # Attention (QKV proj + causal SDPA + out proj) runs on the rank's OWN
    # sequences — identical work in EVERY mode (every scheme shards attention
    # by tokens), so it is a mode-flat dilution term, like the router.
    ATTN_SEQ = 20000  # e2e workload class
    attn_w = None
    if scope == "layer":
        attn_w = [(torch.randn(K, K, dtype=torch.bfloat16) * 0.02).to("cuda")
                  for _ in range(4)]  # q, k, v, o projections
    results = []
    for ci, case in enumerate(plan["cases"]):
        counts = case["counts"]
        m_total = sum(counts)
        a_full = torch.randn(m_total, K, device=dev, dtype=torch.bfloat16)
        outs = {name: torch.empty(m_total, n_out, device=dev, dtype=torch.bfloat16)
                for name, _, _, n_out in gemm_stages}
        h_full = tok_src = out_acc = w_rt = src_tok = w_slot = None
        if scope in ("experts", "moe", "layer"):
            h_full = torch.empty(m_total, N, device=dev, dtype=torch.bfloat16)
        if scope in ("moe", "layer"):
            n_tok = max(1, m_total // TOPK)
            tok_src = torch.randn(n_tok, K, device=dev, dtype=torch.bfloat16)
            out_acc = torch.zeros(n_tok, K, device=dev, dtype=torch.bfloat16)
            w_rt = torch.randn(E, K, device=dev, dtype=torch.bfloat16) * 0.02
            src_tok = torch.randint(0, n_tok, (m_total,), device=dev)
            w_slot = torch.rand(m_total, 1, device=dev, dtype=torch.bfloat16)
            if SHARED_N > 0 and ws_gu is None:
                ws_gu = torch.randn(2 * SHARED_N, K, device=dev, dtype=torch.bfloat16) * 0.02
                ws_down = torch.randn(K, SHARED_N, device=dev, dtype=torch.bfloat16) * 0.02
        counters = base[ALIGN * (1 + ci): ALIGN * (1 + ci) + 144].view(torch.int32)

        own_lo, own_hi = (0, E // 2) if rank == 0 else (E // 2, E)
        _stages_for_scope = gemm_stages[:1] if scope == "gemm" else gemm_stages
        _n_blk_min = min((n_out + 63) // 64 for _, _, _, n_out in _stages_for_scope)
        _pieces = max(1, -(-296 // _n_blk_min))
        segs_all = chunk_segments(counts, _n_blk_min, 4 * N)

        # grid-aware chunking: the contiguous kernel parallelizes over
        # (n_block, segment) and walks m serially per CTA, so a dominant segment
        # gets only n_blk CTAs. Narrow banks (q3: 12 n-blocks) need m-chunking to
        # fill the SMs; wide banks (scout gate_up: 256) never do — chunking them
        # only re-streams multi-hundred-MB banks for nothing. _pieces above is the
        # shared grid-aware chunk count (~2 waves / n_blk_min).

        def _chunk_local(segs):
            # intra-rank hot-segment sub-tiling — an execution courtesy available to
            # ANY system (a kernel-shape artifact cure, not an assignment cost —
            # owned rows must not be charged for it).
            rows = [hi - lo for _, lo, hi in segs]
            if not rows:
                return segs
            avg = max(1, sum(rows) // len(rows))
            # a short list of narrow-bank segments cannot fill the grid at all —
            # chunk unconditionally there; otherwise chunk only the hot outliers.
            force = len(segs) < 24 and len(segs) * _n_blk_min < 296
            out = []
            for e, lo, hi in segs:
                c = hi - lo
                if c > 2 * avg or (force and c > HOT_CHUNK):
                    step = max(-(-c // _pieces), HOT_CHUNK)
                    step = ((step + 127) // 128) * 128  # BLOCK_M-friendly
                else:
                    step = c
                while c > 0:
                    take = min(c, max(step, 1))
                    out.append((e, lo, lo + take))
                    lo += take
                    c -= take
            return out

        seg_sets = {
            "owned": _chunk_local([s for s in segs_all if own_lo <= s[0] < own_hi]),
            "owned_smart": _chunk_local(owned_smart_segments(counts, rank)),
            "sdp": _chunk_local(sdp_segments(counts, rank)),
            "plan": _chunk_local(sep_planner_segments(counts, rank)),
            "queue": segs_all,
        }

        def run(mode: str) -> dict:
            segs = seg_sets[mode]
            off, ids, ls = meta_tensors(segs, dev)
            # elementwise/gather/combine row runs: executed rows for placed modes,
            # the rank's half for queue (the queue splits work near-evenly).
            if mode == "queue":
                half = m_total // 2
                runs = [[0, half]] if rank == 0 else [[half, m_total]]
            else:
                runs = merged_runs(segs)

            def gemm(stage: int, a, b_pin, d):
                if mode == "queue":
                    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(
                        a, b_pin, d, off, ids, ls, counters[3 * stage: 3 * stage + 3],
                        rank, "nk", False)
                elif ls > 1:
                    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                        a, b_pin, d, off, ids, ls, "nk", False)

            busys = []
            for rep in range(args.reps + 1):  # rep 0 = JIT warm, dropped
                if rank == 0:
                    counters.zero_()
                barrier(args.tag, f"c{ci}_{mode}_{rep}a", rank, 2)
                torch.cuda.synchronize()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                if scope == "layer" and attn_w is not None:
                    # attention on the rank's OWN half of tokens: ATTN_SEQ-length
                    # sequences, QKV proj -> causal SDPA -> out proj. Mode-flat.
                    n_tok_l = tok_src.shape[0]
                    lo_a, hi_a = (0, n_tok_l // 2) if rank == 0 else (n_tok_l // 2, n_tok_l)
                    toks = tok_src[lo_a:hi_a]
                    n_seq = toks.shape[0] // ATTN_SEQ
                    # chunk by 4 sequences: same FLOPs, bounded working set
                    # (scout layer scope OOM'd at whole-batch attention tensors)
                    for s0 in range(0, n_seq, 4):
                        ns = min(4, n_seq - s0)
                        x = toks[s0 * ATTN_SEQ:(s0 + ns) * ATTN_SEQ]
                        q = (x @ attn_w[0]).view(ns, ATTN_SEQ, -1, 128).transpose(1, 2)
                        k_ = (x @ attn_w[1]).view(ns, ATTN_SEQ, -1, 128).transpose(1, 2)
                        v = (x @ attn_w[2]).view(ns, ATTN_SEQ, -1, 128).transpose(1, 2)
                        o = torch.nn.functional.scaled_dot_product_attention(
                            q, k_, v, is_causal=True)
                        (o.transpose(1, 2).reshape(ns * ATTN_SEQ, -1) @ attn_w[3])
                        del q, k_, v, o
                if scope in ("moe", "layer"):
                    # router on the rank's OWN half of tokens (identical cost on both
                    # ranks — data-parallel routing precedes any dispatch everywhere)
                    n_tok = tok_src.shape[0]
                    lo_t, hi_t = (0, n_tok // 2) if rank == 0 else (n_tok // 2, n_tok)
                    logits = tok_src[lo_t:hi_t] @ w_rt.t()
                    logits.softmax(dim=-1).topk(min(TOPK, E), dim=-1)
                    for lo, hi in runs:  # gather: pack tokens into the executed rows
                        a_full[lo:hi] = tok_src.index_select(0, src_tok[lo:hi])
                # routed-expert GEMM pipeline (stage 0 only for scope=gemm)
                st0_name, st0_bank, _, _ = gemm_stages[0]
                gemm(0, a_full, st0_bank, outs[st0_name])
                if scope in ("experts", "moe", "layer"):
                    if FUSED_GATEUP:
                        gu = outs["gate_up"]
                        for lo, hi in runs:
                            h_full[lo:hi] = F.silu(gu[lo:hi, :N]) * gu[lo:hi, N:]
                    else:
                        gemm(1, a_full, gemm_stages[1][1], outs["up"])
                        for lo, hi in runs:
                            h_full[lo:hi] = F.silu(outs["gate"][lo:hi]) * outs["up"][lo:hi]
                    dn = len(gemm_stages) - 1
                    gemm(dn, h_full, gemm_stages[dn][1], outs["down"])
                if scope in ("moe", "layer"):
                    if SHARED_N > 0:
                        # shared expert on the rank's OWN half of tokens (mode-flat:
                        # every system runs it data-parallel, no dispatch involved);
                        # two chunks keep the (rows, 2*SHARED_N) temp off the peak.
                        mid = (lo_t + hi_t) // 2
                        for slo, shi in ((lo_t, mid), (mid, hi_t)):
                            su = tok_src[slo:shi] @ ws_gu.t()
                            (F.silu(su[:, :SHARED_N]) * su[:, SHARED_N:]) @ ws_down.t()
                    # combine back to token order: m/2 rows per rank in EVERY mode
                    # (post-return combine is balanced in all real systems);
                    # scale + scatter-write, not bf16 index_add (CAS artifact).
                    lo_c, hi_c = (0, m_total // 2) if rank == 0 else (m_total // 2, m_total)
                    out_acc.index_copy_(0, src_tok[lo_c:hi_c],
                                        outs["down"][lo_c:hi_c] * w_slot[lo_c:hi_c])
                ev1.record()
                torch.cuda.synchronize()
                busys.append(ev0.elapsed_time(ev1) / 1e3)
                barrier(args.tag, f"c{ci}_{mode}_{rep}b", rank, 2)
            out: dict = {"busy_s": sum(busys[1:]) / max(1, len(busys) - 1)}
            n_stages = 1 if scope == "gemm" else len(gemm_stages)
            if mode == "queue":
                # counters/stage: [claimed CTA tickets (grid-sized, > items), head,
                # tail]; executed items = head + tail; per item the kernel streams one
                # (segment, n_block) B slice = stage_bank/n_blk bytes (BLOCK_N=64).
                total_b, ok = 0, True
                for st in range(n_stages):
                    _, _, k_in, n_out = gemm_stages[st]
                    head, tail = int(counters[3 * st + 1]), int(counters[3 * st + 2])
                    n_blk = max(1, (n_out + 63) // 64)
                    mine = head if rank == 0 else tail
                    if head + tail <= 0:
                        ok = False
                    total_b += int(mine * (n_out * k_in * 2) / n_blk)
                out["b_bytes"] = total_b if ok else -1
                out["experts"] = -1  # interleaved; per-side expert set unknown
            else:
                b, ex = analytic_b_bytes(segs)
                stage_bytes = sum(n_out * k_in * 2 for _, _, k_in, n_out in gemm_stages[:n_stages])
                out["b_bytes"] = b * stage_bytes // BANK_BYTES
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
    modes = args.modes.split(",")
    cases = []
    toks = [t.strip() for t in args.alphas.split(",")]
    if any(t == "natural" or t.startswith("g") or (t[0].isdigit() and not t.startswith("z")) for t in toks):
        picks = load_layer_counts(args.hist, args.layers.split(","))
    else:
        picks = {}
    for tok in toks:
        if tok.startswith("z"):  # C1 synthetic zipf: z<s>, one case per ID-shuffle seed
            zs = float(tok[1:])
            for seed in range(args.seeds):
                cases.append({
                    "name": f"zipf{zs}|seed{seed}",
                    "alpha": 0.0,
                    "counts": zipf_counts(zs, args.m_total, seed),
                })
            continue
        for lname, counts in picks.items():
            assert len(counts) == E, (
                f"hist has {len(counts)} experts but --geom says E={E} — real-routing "
                f"columns need a capture from the SAME model")
            if tok == "natural":
                a, g, tag = 0.0, 1.0, "nat"
            elif tok.startswith("g"):   # gamma variant (appendix)
                a, g, tag = 0.0, float(tok[1:]), tok
            else:                        # legacy one-hot alpha (appendix)
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
             "--modes", args.modes, "--scope", args.scope,
             "--geom", args.geom, "--topk", str(args.topk),
             "--fused-gateup", str(int(args.fused_gateup)),
             "--shared-n", str(args.shared_n)],
            env=env, cwd=REPO))
    rcs = [p.wait() for p in procs]
    out = {"m_total": args.m_total, "modes": modes, "scope": args.scope,
           "geom": args.geom, "topk": args.topk, "fused_gateup": args.fused_gateup,
           "shared_n": args.shared_n, "cases": [], "child_rcs": rcs}
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
    ap.add_argument("--modes", default="owned,sdp,plan,queue")
    ap.add_argument("--scope", default="gemm", choices=["gemm", "experts", "moe", "layer"])
    ap.add_argument("--geom", default="128,768,2048", help="E,N,K (default q3-30b)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--fused-gateup", type=lambda v: bool(int(v)), default=False)
    ap.add_argument("--shared-n", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    # NAMING EPOCH 2026-07-10: bench mode "sep" renamed "plan" (matches the e2e
    # backend asym_sepplan2); the old spelling is accepted and mapped.
    args.modes = ",".join("plan" if m == "sep" else m for m in args.modes.split(","))
    set_geometry(args.geom, args.topk, args.fused_gateup, args.shared_n)
    if args.role == "child":
        return child_main(args)
    if any(t.strip() == "natural" or t.strip().startswith("g") or
           (t.strip()[0].isdigit()) for t in args.alphas.split(",")):
        assert args.hist, "--hist required for natural/gamma/alpha tokens (capture via ASYM_EP_STATS=1)"
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
