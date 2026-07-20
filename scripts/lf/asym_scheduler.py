#!/usr/bin/env python3
"""asym_scheduler — the allocation-form workload scheduler (core system design).

Formulation (agent/handoffs/prompt.md v2, scheduler_v2.md §9): HBM is a fixed
endowment, always fully spendable; the scheduler is a WATER-FILL ALLOCATOR that
buys, one unit at a time, whichever asset has the highest marginal Δtok/s per GiB:
  asset 1: batch units (+1 sample; return collapses at the knee N*)
  asset 2: residency knobs, per family, in measured ρ order (staged → ker000 →
           keep-acts → panel-cache); shedding order is the reverse.
"Latency mode" / "memory mode" are LABELS of the emitted portfolio, never inputs.
User-facing inputs: model, seq (+ optional safety ∈ {conservative,normal,aggressive},
reserved_hbm GiB). Batch, family, knobs, mode label: OUTPUTS.

Constants: fitted from the c14 campaign (agent/impls/s04-p1-dgx-02-c14/,
scheduler_v2.md §3b/§7/§9.2). Self-calibration: probe-vs-predict mismatch > tol
⇒ refit that family's 4 constants from 2 probes (not automated here; see §8).

CLI:  asym_scheduler.py MODEL SEQ [--safety normal] [--reserved 0] [--sweep]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace

GIB = 1.0  # all memory numbers in GiB

# ── hardware ────────────────────────────────────────────────────────────────
C_PHYS = 185.0          # GB200 HBM (189471 MiB)
BEND_GIB = 4.0          # near-wall allocator bend: linear model over-predicts by 3-5
EDGE_PENALTY = {0.92: 0.00, 0.95: 0.01, 0.98: 0.04}  # measured 0-6%; keyed by util ceiling
SAFETY_H = {"conservative": 0.08, "normal": 0.05, "aggressive": 0.02}

# ── measured model constants (q3-30b-a3b; per-1k-seq units) ─────────────────
# tau(s) = a + b*s_k  [us/tok, b1 deep-end fits]   M(s,B) = base + B*m*s_k [GiB]
@dataclass(frozen=True)
class Family:
    name: str
    a: float            # per-token intercept us/tok
    b: float            # per-token slope us/tok per 1k seq
    base: float         # GiB intercept
    m: float            # GiB per sample per 1k seq (leanest rung of this family)
    rungs: tuple = ()   # residency knobs, ρ-DESCENDING (admit order); shed reverse

@dataclass(frozen=True)
class Rung:
    name: str
    dm_fixed: float     # GiB, batch/seq-independent part
    dm_per_ktok: float  # GiB per 1k TOKENS (B*s_k)
    dtau: float         # us/tok SAVED while admitted (measured A/B deltas)
    env: dict = field(default_factory=dict)

# q3-30b (MoE). asym rungs measured: staged −70us/tok, ker000 −34, keep-acts −33
# (120k×8 ladder per-token); Phase-2 fixes folded into the lat intercept a=126.
# Slope decomposition check: mem 0.100 + ker000 0.0375 + KA 0.038 ≈ lat 0.179 ✓ (±bend)
ASYM = Family(
    "asym", a=126.0, b=1.937, base=4.3, m=0.100,  # m = leanest rung (all shed, streamed engine)
    rungs=(
        Rung("staged",  2.0, 0.0,    70.0, {"ASYM_GEMM_DISPATCH": "staged"}),
        Rung("ker000",  0.0, 0.0375, 34.0, {"_KER": "ker000"}),  # else ker101
        # 2026-07-19 GPU calibration (tputsched 900k): keep-acts memory grows SUPERLINEARLY
        # past ~800k (measured 183.0 GiB @900k vs 170.2 predicted; slope 0.295/1k on 800->900k
        # vs 0.179 fit). dm_per_ktok raised to the 800-900k secant; shed boundary moves left
        # (~900k). Predicted tok/s validated: 519 measured vs 535 (-3.0%).
        Rung("keep-acts", 0.0, 0.052, 33.0, {
            "ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM": "1", "ASYMM_ATTN_ACT_KEEP_ACTS_HBM": "1",
            "ASYM_GC_SAVE_ON_CPU_OVERRIDE": "false"}),
        Rung("panel-cache", 6.0, 0.0, 3.0, {"ASYM_W_PANEL_CACHE_GB": "6"}),
    ),
)
SUP_UNS = Family("sup-unsloth", a=-63.0, b=2.236, base=19.9, m=0.253)
SUP_RC  = Family("sup-recomp",  a=62.0,  b=1.995, base=4.1,  m=0.452)
FAMILIES = (ASYM, SUP_UNS, SUP_RC)
KNEE_TOKENS_K = 400.0   # N* ≈ 0.4M tokens (measured batch-flat above)
NOISE = 0.015           # run-to-run tok/s noise for tie-breaks
FIT_RANGE_K = (160.0, 800.0)   # deep-end b1 fit validity; below → anchor lookup, above → extrapolation (flagged)
# measured big-batch peaks (c12/c14 tables) — SHORT-SEQ REGIME TRUTH, not derivable from
# the deep-end fits (different MFU regime): {seq_k: {family: (B, tok/s, GiB)}}
ANCHORS = {
    64.0:  {"sup-unsloth": (8, 5883, 150.0), "sup-recomp": (6, 5919, 181.0), "asym": (8, 4200, 150.0)},
    80.0:  {"sup-unsloth": (8, 3424, 176.9), "asym": (8, 3642, 84.7)},
    96.0:  {"sup-unsloth": (6, 2900, 160.0), "asym": (8, 3200, 100.0)},
    128.0: {"sup-unsloth": (4, 3055, 149.9), "sup-recomp": (2, 2985, 119.5), "asym": (8, 2723, 126.7)},
}

# class-1 pins always emitted with asym (never dialed):
ASYM_PINS = {
    "ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU": "1", "ASYMM_QWEN3_MOE_FG_DA_GPU": "1",
    "ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS": "0",
    "ASYMM_FG_ELEMENTWISE_CHUNK_MB": "1024", "ASYMM_QWEN3_MOE_DOWN_DX_STAGED": "1",
    "ASYMM_FUSED_LORA_ADDMM": "1", "ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X": "1",
}


@dataclass
class Plan:
    family: str; B: int; rungs: list; mode: str
    tok_s: float; mem: float; util: float; env: dict


def _mem(fam: Family, s_k: float, B: int, admitted) -> float:
    m = fam.base + B * fam.m * s_k
    for r in admitted:
        m += r.dm_fixed + r.dm_per_ktok * B * s_k
    return m


def _tau(fam: Family, s_k: float, admitted) -> float:
    t = fam.a + fam.b * s_k
    shed = [r for r in fam.rungs if r not in admitted]
    return t + sum(r.dtau for r in shed)   # a is calibrated with ALL rungs on


def _toks(fam: Family, s_k: float, B: int, admitted, ceff: float) -> float:
    mem = _mem(fam, s_k, B, admitted)
    util = mem / C_PHYS
    if mem > ceff + BEND_GIB:
        return -1.0
    pen = 0.0
    for cap, p in sorted(EDGE_PENALTY.items()):
        if util > cap: pen = p
    return 1e6 / _tau(fam, s_k, admitted) * B / max(B, 1) / (1 + pen)  # tok/s per-token basis ×1 (B cancels above knee)


def plan_family(fam: Family, s_k: float, ceff: float) -> Plan | None:
    """Water-fill: start bare (B=1, no rungs); buy highest-marginal-return unit."""
    admitted: list = []
    B = 1
    if _mem(fam, s_k, B, admitted) > ceff + BEND_GIB:
        return None
    while True:
        best = None  # (gain_per_gib, kind, payload)
        cur = _toks(fam, s_k, B, admitted, ceff)
        # asset 1: +1 batch unit (return ≈ 0 above the knee — c_fix amortization only)
        nb = B + 1
        if _mem(fam, s_k, nb, admitted) <= ceff:
            below_knee = (B * s_k) < KNEE_TOKENS_K
            gain = cur * 0.9 if below_knee else cur * 0.001   # knee law
            dm = _mem(fam, s_k, nb, admitted) - _mem(fam, s_k, B, admitted)
            best = _better(best, (gain / dm, "batch", nb))
        # asset 2: next un-admitted rung (ρ order enforced by list order)
        for r in fam.rungs:
            if r in admitted: continue
            trial = admitted + [r]
            if _mem(fam, s_k, B, trial) > ceff: break  # ladder truncates; later rungs cost ≥
            gain = _toks(fam, s_k, B, trial, ceff) - cur
            dm = max(_mem(fam, s_k, B, trial) - _mem(fam, s_k, B, admitted), 1e-6)
            best = _better(best, (gain / dm, "rung", r))
            break  # only the NEXT rung is purchasable (nested order)
        if best is None or best[0] <= 0:
            break
        if best[1] == "batch": B = best[2]
        else: admitted.append(best[2])
    mem = _mem(fam, s_k, B, admitted)
    names = [r.name for r in admitted]
    mode = ("latency" if len(names) >= 3 else
            "balanced" if "staged" in names else "memory")
    env = dict(ASYM_PINS) if fam.name == "asym" else {}
    for r in admitted: env.update(r.env)
    return Plan(fam.name, B, names, mode if fam.name == "asym" else "-",
                _toks(fam, s_k, B, admitted, ceff), mem, mem / C_PHYS, env)


def _better(cur, cand):
    return cand if cur is None or cand[0] > cur[0] else cur


def schedule(s_k: float, safety="normal", reserved=0.0) -> Plan:
    ceff = (C_PHYS - reserved) * (1 - SAFETY_H[safety])
    plans = [p for p in (plan_family(f, s_k, ceff) for f in FAMILIES) if p]
    if not plans:
        raise SystemExit(f"no family feasible at s={s_k:.0f}k (reserved={reserved})")
    plans.sort(key=lambda p: -p.tok_s)
    top = plans[0]
    for p in plans[1:]:  # tie-break to leaner within noise
        if p.tok_s >= top.tok_s * (1 - NOISE) and p.mem < top.mem:
            top = p
    return top


def _selftest() -> int:
    """Cleanness assertions — the design test. Exit nonzero on violation."""
    fails = 0
    # 1. NESTED SHEDDING along seq: asym rung set must only lose rungs (suffix-shed) as s grows
    seqs = list(range(640, 1520, 20))
    prev = None
    for sk in seqs:
        p = plan_family(ASYM, float(sk), C_PHYS * (1 - SAFETY_H["normal"]))
        cur = tuple(p.rungs)
        if prev is not None:
            if not (len(cur) <= len(prev) and list(cur) == [r for r in prev if r in cur]):
                print(f"FAIL nestedness at {sk}k: {prev} -> {cur}"); fails += 1
            rev = [r.name for r in reversed(ASYM.rungs)]   # shed order = reverse-admit (reverse-ρ)
            shed_prev = [r for r in rev if r not in prev]
            shed_cur = [r for r in rev if r not in cur]
            if shed_cur[:len(shed_prev)] != shed_prev:
                print(f"FAIL shed-order at {sk}k: shed so far {shed_cur} vs prior {shed_prev}"); fails += 1
        prev = cur
    # 2. tok/s MONOTONE DECREASING in s (no spurious speedups from shedding)
    last = None
    for sk in seqs:
        t = plan_family(ASYM, float(sk), C_PHYS * (1 - SAFETY_H["normal"])).tok_s
        if last is not None and t > last * 1.001:
            print(f"FAIL monotone tok/s at {sk}k: {last:.0f} -> {t:.0f}"); fails += 1
        last = t
    # 3. RESERVED-HBM sweep at fixed s: rung set shrinks monotonically with reservation
    prev = None
    for res in range(0, 120, 10):
        p = plan_family(ASYM, 800.0, (C_PHYS - res) * (1 - SAFETY_H["normal"]))
        if p is None: break
        cur = tuple(p.rungs)
        if prev is not None and not (len(cur) <= len(prev) and list(cur) == [r for r in prev if r in cur]):
            print(f"FAIL reserved-nestedness at res={res}: {prev} -> {cur}"); fails += 1
        prev = cur
    # 4. BOUNDARY consistency: keep-acts shed point must match the ANALYTIC crossing derived
    # from the SAME constants (self-consistent under refits — the learning loop moves both).
    ceff = C_PHYS * (1 - SAFETY_H["normal"])
    slope = ASYM.m + sum(r.dm_per_ktok for r in ASYM.rungs if r.name in ("ker000", "keep-acts"))
    fixed = ASYM.base + sum(r.dm_fixed for r in ASYM.rungs if r.name in ("staged",))
    analytic = (ceff - fixed) / slope
    shed_at = None
    for sk in seqs:
        p = plan_family(ASYM, float(sk), ceff)
        if "keep-acts" not in p.rungs: shed_at = sk; break
    if shed_at is None or abs(shed_at - analytic) > 60:
        print(f"FAIL boundary: keep-acts shed at {shed_at}k, analytic {analytic:.0f}k"); fails += 1
    # 5. SAFETY dial: aggressive must never emit less than conservative admits
    for sk in (900.0, 1000.0, 1200.0):
        pa = plan_family(ASYM, sk, C_PHYS * (1 - SAFETY_H["aggressive"]))
        pc = plan_family(ASYM, sk, C_PHYS * (1 - SAFETY_H["conservative"]))
        if len(pa.rungs) < len(pc.rungs):
            print(f"FAIL safety-monotone at {sk}k"); fails += 1
    print("SELFTEST", "PASS (5/5 properties)" if fails == 0 else f"{fails} FAILURES")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="q3-30b-a3b")
    ap.add_argument("seq", nargs="?", type=float, default=480000)
    ap.add_argument("--safety", default="normal", choices=list(SAFETY_H))
    ap.add_argument("--reserved", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.model != "q3-30b-a3b":
        print(f"[warn] constants are fitted for q3-30b-a3b; {a.model} needs a 2-probe refit", file=sys.stderr)
    if a.sweep:
        print(f"{'seq':>7} | {'family':<11} {'B':>2} {'mode':<8} {'rungs':<38} {'pred tok/s':>10} {'GiB':>6} {'util':>5}")
        for sk in (64, 80, 96, 128, 160, 208, 256, 320, 400, 480, 560, 640, 720, 800, 880, 940, 990, 1040, 1100, 1200, 1300, 1400, 1500):
            if float(sk) in ANCHORS:  # short-seq regime: measured truth, not fits
                best = max(ANCHORS[float(sk)].items(), key=lambda kv: kv[1][1])
                fam, (B, ts, gib) = best
                print(f"{sk:>6}k | {fam:<11} {B:>2} {'anchor':<8} {'(measured big-batch regime)':<38} {ts:>10.0f} {gib:>6.1f} {gib/C_PHYS:>5.0%}")
                continue
            p = schedule(sk, a.safety, a.reserved)
            tag = "" if FIT_RANGE_K[0] <= sk <= FIT_RANGE_K[1] else "*"
            print(f"{sk:>6}k | {p.family:<11} {p.B:>2} {p.mode:<8} {','.join(p.rungs):<38} {p.tok_s:>9.0f}{tag:<1} {p.mem:>6.1f} {p.util:>5.0%}")
        print("(* = extrapolated beyond the fitted range — exactly what the ultra-long GPU runs validate)")
        return
    p = schedule(a.seq / 1000, a.safety, a.reserved)
    print(f"EMIT family={p.family} B={p.B} mode={p.mode} rungs={p.rungs}")
    print(f"     predicted {p.tok_s:.0f} tok/s | {p.mem:.1f} GiB ({p.util:.0%} of device)")
    for k, v in sorted(p.env.items()):
        print(f"     env {k}={v}")


if __name__ == "__main__":
    main()
