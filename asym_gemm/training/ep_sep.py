"""S6 TRUE sEP (fix_gb200_ep.md S6): union expert-work sharing over the shared
pinned fabric. TWO flavors share every line of transport (NAMING EPOCH 2026-07-10):
  queue (asym_sepqueue2, ASYM_EP_SEP_MODE=queue, default) — side 0 pops the union
        front, side 1 the back, on a SHARED counter block; the meet point IS the
        split (dynamic, needs no counts).
  plan  (asym_sepplan2, ASYM_EP_SEP_MODE=plan) — the split is COMPUTED from the
        union counts (contiguous cut at a segment boundary, both ranks derive it
        identically); each side launches only its sublist with a PRIVATE counter
        block, then fabricates the meet point so spin_gather is unchanged.

Per armed grouped launch, BOTH ranks execute ONE union work list (side 0 pops from
the front, side 1 from the back — the ep_steal kernel family):
  union = [rank0's segments | rank1's segments]   (pairs offsets, each in its
           OWNER's padded-d row coordinates; n_own = rank0's segment count)
  a       = my local packed A;  a_peer = peer's packed A staged in pinned memory
  d       = my local padded D;  d_peer = MY staging mirroring the PEER's D (rows I
            steal land there at the peer's row offsets; the peer spin-gathers them)
TRANSPORT IS COLLECTIVE-FREE: cross-rank coupling is (i) host-spins on pinned
header flags (host<->host only, never drains a CUDA stream => the vanilla-EP
stagger class is structurally absent) and (ii) GPU release/acquire flags
(ep_steal_flag_set / spin_gather early-exit). Weights NEVER move: banks stream
from the shared fabric exactly as sdp2; the union merely decides WHICH rank's
kernels stream WHICH banks (bank-once consolidation — the S6 streaming win).

ARMING RULE (the sdp-floor): a launch arms only when m/(list_size-1) <=
ASYM_EP_SEP_MAX_MPE (default 4096 rows/segment — the measured streaming-bound
boundary); un-armed launches fall through to the caller's plain/queued path, so
sdp2 behavior is the guaranteed floor.

LoRA/grad semantics are UNTOUCHED: stealing covers only the frozen grouped GEMM;
every rank's LoRA branch and backward run on its OWN rows exactly as sdp2.

Buffers are injectable (install_buffers) so the standalone probe can validate the
protocol on plain pinned allocations without the fabric.
"""
from __future__ import annotations

import os
import time

import torch

_ENABLED = os.environ.get("ASYM_EP_SEP") == "1"
_MAX_MPE = int(os.environ.get("ASYM_EP_SEP_MAX_MPE", "4096") or 4096)
_SPIN_TIMEOUT_S = float(os.environ.get("ASYM_EP_SEP_SPIN_TIMEOUT_S", "120") or 120)
# NAMING EPOCH 2026-07-10 (fix_gb200_ep_v2): backend asym_sepqueue2 -> mode "queue"
# (counter-raced steal — the original S6 flavor); asym_sepplan2 -> mode "plan"
# (same union/transport, the cut COMPUTED from counts; no racing).
_MODE = os.environ.get("ASYM_EP_SEP_MODE", "queue") or "queue"
if _ENABLED and _MODE not in ("queue", "plan"):
    raise RuntimeError(f"ASYM_EP_SEP_MODE must be 'queue' or 'plan', got '{_MODE}'")

RING = 4
FLAG_SLOTS = 256          # rotating hdr/done flag indices; far-ahead host zeroing
MAX_SEGS = 2048           # per rank per launch (E + hot chunks headroom)
_HDR_INTS = 2 + 3 * MAX_SEGS  # [m_padded, n_segs, (expert, lo, hi) * MAX_SEGS]


def enabled() -> bool:
    return _ENABLED


class _SepState:
    """Protocol state over injected pinned buffers (fabric-backed in production)."""

    def __init__(self, *, rank: int, world: int, ctrl: torch.Tensor,
                 x_slots: list[list[torch.Tensor]], d_slots: list[list[torch.Tensor]],
                 side_stream: torch.cuda.Stream | None = None) -> None:
        assert world == 2, "S6 true-sEP v1 is EP=2 only"
        assert ctrl.is_pinned() and ctrl.dtype == torch.int32
        self.rank, self.world = rank, world
        self.ctrl = ctrl
        # ctrl layout (int32 indices):
        #   [0,               2*FLAG_SLOTS)                       hdr flags   [rank][slot]
        #   [2*FLAG_SLOTS,    4*FLAG_SLOTS)                       done flags  [rank][slot]
        #   [4*FLAG_SLOTS,    4*FLAG_SLOTS + RING*8)              queue counter blocks
        #   [.. + RING*8,     .. + RING*8 + 2*RING*_HDR_INTS)     headers [rank][ring]
        need = 4 * FLAG_SLOTS + RING * 8 + 2 * RING * _HDR_INTS
        assert ctrl.numel() >= need, (ctrl.numel(), need)
        self._hdr_flag_base = 0
        self._done_flag_base = 2 * FLAG_SLOTS
        self._ctr_base = 4 * FLAG_SLOTS
        self._hdr_base = self._ctr_base + RING * 8
        self.x_slots = x_slots  # x_slots[rank][ring] flat pinned bf16
        self.d_slots = d_slots  # d_slots[rank][ring] flat pinned bf16 (my staging OF PEER rows)
        self.seq = 0
        self.mode = _MODE
        if self.mode == "plan":
            # the steal kernel is ONE ITEM PER CTA; queue mode tolerates the tuned
            # 75% grid because the two sides' grids overlap on the SHARED counters,
            # but a private-block sublist launch must cover ALL its items alone —
            # probe receipt: bad rows began exactly at grid_y_local * n_blk.
            prior = os.environ.get("DG_EP_QUEUE_GRID_PCT")
            if prior not in (None, "", "100"):
                print(f"[ep_sep] plan mode forces DG_EP_QUEUE_GRID_PCT=100 (was {prior})")
            os.environ["DG_EP_QUEUE_GRID_PCT"] = "100"
        self.side_stream = side_stream or torch.cuda.Stream()
        self.stats = {"armed": 0, "declined": 0, "spin_wait_s": 0.0}

    # ---- ctrl accessors -------------------------------------------------------
    def _hdr_flag_idx(self, rank: int, seq: int) -> int:
        return self._hdr_flag_base + rank * FLAG_SLOTS + (seq % FLAG_SLOTS)

    def _done_flag_idx(self, rank: int, seq: int) -> int:
        return self._done_flag_base + rank * FLAG_SLOTS + (seq % FLAG_SLOTS)

    def _counters(self, seq: int, block: int = 0) -> torch.Tensor:
        # block 0 = the shared queue-race block (both sides). Plan mode gives each
        # side a PRIVATE block (0/1) inside the same ring slot — 6 of the 8 ints.
        lo = self._ctr_base + (seq % RING) * 8 + 3 * block
        return self.ctrl[lo:lo + 3]

    def _header(self, rank: int, seq: int) -> torch.Tensor:
        lo = self._hdr_base + (rank * RING + (seq % RING)) * _HDR_INTS
        return self.ctrl[lo:lo + _HDR_INTS]

    # ---- the armed launch -----------------------------------------------------
    def pre_gate(self, m: int, k: int, n_segs: int) -> bool:
        """Host-int decline check BEFORE the caller pays offsets.cpu().tolist()
        (a per-call GPU sync — measured +6.7 s/step at 20k where every call
        declines). Declining here consumes the seq and PUBLISHES flag=2 exactly
        like try_armed's decline (asymmetric peers resolve via the published
        flag); passing consumes nothing — try_armed follows and owns the seq."""
        x_slot = self.x_slots[self.rank][self.seq % RING]
        decline = (
            n_segs == 0 or n_segs > MAX_SEGS or m == 0
            or m / max(1, n_segs) > _MAX_MPE
            or m * k > x_slot.numel()
        )
        if not decline:
            return True
        seq = self.seq
        self.seq += 1
        ahead = seq + FLAG_SLOTS // 2
        self.ctrl[self._hdr_flag_idx(self.rank, ahead)] = 0
        self.ctrl[self._done_flag_idx(self.rank, ahead)] = 0
        self.stats["declined"] += 1
        self.ctrl[self._hdr_flag_idx(self.rank, seq)] = 2
        return False

    def try_armed(self, asym_gemm, a: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                  segs: list[tuple[int, int, int]], compiled_dims: str,
                  transpose_b: bool) -> bool:
        """segs: my (expert, lo, hi) in MY padded-d coordinates. Returns True when the
        union launch handled this GEMM (d filled, incl. gathered stolen rows)."""
        m, k = int(a.shape[0]), int(a.shape[1])
        n = int(d.shape[1])
        n_segs = len(segs)
        # every eligible call site consumes ONE seq on BOTH ranks (launch alignment);
        # a decline is PUBLISHED (host-written flag=2) so the peer never deadlocks —
        # if EITHER side declines, BOTH fall back to their local path.
        seq = self.seq
        self.seq += 1
        ring = seq % RING
        me, peer = self.rank, 1 - self.rank
        x_slot = self.x_slots[me][ring]
        d_stage_mine = self.d_slots[me][ring]      # I write PEER rows here

        # far-ahead flag hygiene (host writes; slots FLAG_SLOTS/2 ahead are idle)
        ahead = seq + FLAG_SLOTS // 2
        self.ctrl[self._hdr_flag_idx(me, ahead)] = 0
        self.ctrl[self._done_flag_idx(me, ahead)] = 0

        decline = (
            n_segs == 0 or n_segs > MAX_SEGS or m == 0
            or m / max(1, n_segs) > _MAX_MPE          # sdp-floor: compute-bound
            or m * k > x_slot.numel()
        )
        if decline:
            self.stats["declined"] += 1
            self.ctrl[self._hdr_flag_idx(me, seq)] = 2  # published decline
            return False

        # counters zero (rank0, before publishing its header flag => ordered for both)
        if me == 0:
            self._counters(seq).zero_()

        # header: m_padded, n_segs, segs
        hdr = self._header(me, seq)
        hdr[0] = m
        hdr[1] = n_segs
        flat = hdr[2:2 + 3 * n_segs].view(n_segs, 3)
        seg_t = torch.tensor(segs, dtype=torch.int32)
        flat.copy_(seg_t)

        # stage my packed A -> my pinned X slot on the side stream; the hdr flag is
        # published THROUGH THE HOST (same PR-5 visibility receipt as the done flag:
        # GPU-side release can beat the copy's arrival for the PEER GPU's TMA reads).
        # The event wait is a LOCAL drain of the copy only — never peer-coupled.
        main = torch.cuda.current_stream()
        self.side_stream.wait_stream(main)
        _C = getattr(asym_gemm, "_C", asym_gemm)
        with torch.cuda.stream(self.side_stream):
            x_slot[: m * k].view(m, k).copy_(a, non_blocking=True)
            x_ev = torch.cuda.Event()
            x_ev.record()
        x_ev.synchronize()
        self.ctrl[self._hdr_flag_idx(me, seq)] = 1

        # host spin: peer header (host<->host pinned read; no stream interaction)
        t0 = time.perf_counter()
        pf = self._hdr_flag_idx(peer, seq)
        while int(self.ctrl[pf]) == 0:
            if time.perf_counter() - t0 > _SPIN_TIMEOUT_S:
                raise RuntimeError(f"sEP hdr spin timeout (seq {seq}, rank {me})")
        self.stats["spin_wait_s"] += time.perf_counter() - t0
        if int(self.ctrl[pf]) == 2:  # peer declined => symmetric fallback
            self.stats["declined"] += 1
            self.stats["peer_declined"] = self.stats.get("peer_declined", 0) + 1
            return False
        ph = self._header(peer, seq)
        m_peer = int(ph[0])
        n_segs_peer = int(ph[1])
        peer_segs = ph[2:2 + 3 * n_segs_peer].view(n_segs_peer, 3).tolist()

        # union: [rank0 section | rank1 section], each expert-sorted (bank affinity;
        # side 1 pops from the back so its section ends with its highest experts)
        mine_sorted = sorted(segs, key=lambda s: s[0])
        peer_sorted = sorted([tuple(s) for s in peer_segs], key=lambda s: s[0])
        r0_segs, r1_segs = (mine_sorted, peer_sorted) if me == 0 else (peer_sorted, mine_sorted)
        n_own = len(r0_segs)
        union = r0_segs + r1_segs
        total = len(union)
        pairs = torch.tensor([[lo, hi] for _, lo, hi in union], dtype=torch.int32,
                             device=a.device).reshape(-1)
        ids = torch.tensor([e for e, _, _ in union] + [-1], dtype=torch.int32,
                           device=a.device)

        peer_x = self.x_slots[peer][ring][: m_peer * k].view(m_peer, k)
        d_peer = d_stage_mine[: m_peer * n].view(m_peer, n)

        if self.mode == "plan":
            # PLAN flavor (asym_sepplan2): identical union + transport, but the cut
            # is COMPUTED from the counts — a contiguous prefix/suffix split at a
            # segment boundary balancing rows (the arming rule caps rows/segment at
            # _MAX_MPE, so segment granularity is fine). Each side launches ONLY its
            # sublist with a PRIVATE counter block (zero-steal claims it whole; no
            # cross-GPU counter traffic at all); one host write then fabricates the
            # queue-final counter state so the unchanged spin_gather sees the
            # planned meet point. Both ranks compute the same cut from the same
            # union — no extra exchange.
            rows = [hi - lo for _, lo, hi in union]
            acc, half = 0, sum(rows) / 2.0
            k_seg, best_gap = 0, float("inf")
            for i, r in enumerate(rows):
                acc += r
                gap = abs(acc - half)
                if gap < best_gap:
                    best_gap, k_seg = gap, i + 1
            # HYSTERESIS: crossing the ownership boundary couples this launch on the
            # peer's completion (the gather must wait for its done flag). Only pay
            # that when the cut actually moves meaningful work — under equal-shard
            # DP the row-balanced cut lands within a segment of n_own, and snapping
            # to n_own makes the launch fully local (gather early-exits, zero
            # coupling), exactly like an un-stolen queue launch.
            own_rows = sum(rows[:n_own])
            if abs(own_rows - half) <= _MAX_MPE:
                k_seg = n_own
            if 0 < k_seg < total:
                lo_seg, hi_seg = (0, k_seg) if me == 0 else (k_seg, total)
                sub_pairs = pairs[2 * lo_seg: 2 * hi_seg]
                sub_ids = torch.cat([ids[lo_seg:hi_seg], ids[total:total + 1]])
                sub_n_own = max(0, min(n_own, hi_seg) - lo_seg)
                ctr = self._counters(seq, block=me)
                ctr.zero_()
                n_blk = asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal(
                    a, b, d, peer_x, d_peer, sub_pairs, sub_ids,
                    (hi_seg - lo_seg) + 1, ctr, me, sub_n_own,
                    compiled_dims, transpose_b)
                done_ev = torch.cuda.Event()
                done_ev.record()
                done_ev.synchronize()
                self.ctrl[self._done_flag_idx(me, seq)] = 1
                # fabricate the meet point on MY private block (host write happens
                # before the gather kernel launch => visible to its sysmem reads)
                ctr[1] = k_seg * n_blk
                ctr[2] = (total - k_seg) * n_blk
                _C.ep_steal_spin_gather(
                    d, self.d_slots[peer][ring][: m * n].view(m, n), pairs,
                    ctr, self.ctrl, self._done_flag_idx(peer, seq),
                    n_blk, total, n_own, me)
                self.stats["armed"] += 1
                self.stats["planned"] = self.stats.get("planned", 0) + 1
                return True
            # degenerate cut (one side would idle) => fall through to the queue race
            # (both ranks compute the same k_seg, so the branch stays symmetric)

        n_blk = asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal(
            a, b, d, peer_x, d_peer, pairs, ids, total + 1,
            self._counters(seq), me, n_own, compiled_dims, transpose_b)
        # DONE published THROUGH THE HOST (PR-5 race receipt: the GPU-side release
        # flag can beat the final TMA sysmem stores' arrival as observed by the PEER
        # GPU; host observation of kernel completion guarantees global visibility of
        # all its writes). This is a LOCAL drain (own-GPU wait) — never peer-coupled,
        # so it is sdp-benign and cannot re-create the stagger class.
        done_ev = torch.cuda.Event()
        done_ev.record()
        done_ev.synchronize()
        self.ctrl[self._done_flag_idx(me, seq)] = 1
        _C.ep_steal_spin_gather(
            d, self.d_slots[peer][ring][: m * n].view(m, n), pairs,
            self._counters(seq), self.ctrl, self._done_flag_idx(peer, seq),
            n_blk, total, n_own, me)
        self.stats["armed"] += 1
        return True


_STATE: _SepState | None = None


def install_buffers(*, rank: int, world: int, ctrl: torch.Tensor,
                    x_slots: list[list[torch.Tensor]],
                    d_slots: list[list[torch.Tensor]]) -> _SepState:
    global _STATE
    _STATE = _SepState(rank=rank, world=world, ctrl=ctrl, x_slots=x_slots, d_slots=d_slots)

    import atexit

    def _dump_stats(st=_STATE):
        s = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.stats.items()}
        print(f"[ep_sep] rank{st.rank} mode={st.mode} exit stats: {s}", flush=True)

    atexit.register(_dump_stats)
    return _STATE


def state() -> _SepState | None:
    return _STATE


def ctrl_ints_needed() -> int:
    return 4 * FLAG_SLOTS + RING * 8 + 2 * RING * _HDR_INTS
