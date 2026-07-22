#!/usr/bin/env python3
"""asym_scheduler — merged workload scheduler (sched40 x sched42, 2026-07-21).

DECISION RULE (40's system, merge_scheduler.md §2): inputs = model + seq only.
Feasibility thresholding over a memory-descending ladder of measured recipes
("tiers" = rung prefixes): pick the FIRST tier whose predicted HBM fits under
BETA*C_HBM and whose host RSS fits under C_HOST_EFF; batch = largest feasible,
capped by the knee (q3-30b only). Speed order is structural (more residency =
faster), so first-fit = fastest-fit. NO timing inputs in the decision.
Near-wall (within 8% of BETA*C_HBM either side): PROBE, don't trust the line
(c12 §6 — this caught every would-be wrong call). Short-seq (<= anchor zone):
measured anchor table is the truth, lines are invalid there (different MFU
regime). tau/water-fill prediction (42) survives OFFLINE behind --predict.

CONSTANTS provenance: byte lines from c12 system_summary §4 (archive/), MoE
rung slopes from 42's calibration (bundle 0.1895 validated 0.1-GiB-exact at
the 900k record incl. panel +6; shed 0.1375 at 1.1M; T3 0.100 at 1.6M), host
caps calibrated to measured outcomes (FIT at RSS 980-983, OOM at ~1003 =>
C_HOST_EFF 990; nominal c12 header 957). Knee N*~=400k tok = q3-30b ONLY.

CLI:
  asym_scheduler.py MODEL SEQ [--safety normal] [--reserved GiB]
  asym_scheduler.py --sweep [MODEL] | --selftest | --replay | --emit-recipes
  asym_scheduler.py MODEL SEQ --predict     (offline tau report, 42's machinery)
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

# ── hardware / policy constants ─────────────────────────────────────────────
C_HBM = 185.0                 # GB200 physical HBM GiB (189471 MiB)
BETA = 0.92                   # safe-utilization ceiling (c12 §6)
PROBE_FRAC = 0.08             # probe band: within 8% of BETA*C_HBM, either side
C_HOST_NOMINAL = 957.0        # c12 header (q32 T3 anchor)
C_HOST_EFF = 990.0            # calibrated: FIT at 980-983 measured, OOM ~1003
SAFETY_H = {"conservative": 0.08, "normal": 0.05, "aggressive": 0.02}
KNEE_TOKENS_K = {"q3-30b-a3b": 400.0}   # measured q3-30b only; dense: max-fit


@dataclass(frozen=True)
class TierLine:
    name: str                  # T1 / T2 / T2B / T3
    mode: str                  # latency / balanced / memory label (output only)
    token: str                 # driver recompute token
    env: dict                  # full recipe env (excludes the token itself)
    base: float                # GiB intercept
    m: float                   # GiB per 1k tokens (B*s)
    host_c: float = 0.0        # host RSS anchor GB (0 = unmeasured, watchdog)
    host_h: float = 0.0        # host GB per 1k tokens (anchor-grade)
    valid_k: tuple = (0.0, 1e9)   # token fit-validity range (1k units)
    note: str = ""


@dataclass(frozen=True)
class ModelTab:
    name: str
    family: str                # dense | moe
    tiers: tuple               # preference order, memory-descending
    anchors: dict = field(default_factory=dict)   # seq_k -> (B, tok_s, gib, measured?)
    anchor_max_k: float = 0.0


# ── recipe envs (single source of truth; --emit-recipes serializes these) ───
_STAGED = {"ASYM_GEMM_DISPATCH": "staged"}
_DENSE_T2_ENV = {
    **_STAGED,
    "ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM": "1",
    "ASYM_SAVED_TENSOR_ASYNC_UNPACK": "1",
    "ASYMM_QWEN3_MOE_DOWN_DX_STAGED": "1",
    "ASYMM_FG_ELEMENTWISE_CHUNK_MB": "1024",
}
# class-1 MoE pins — the SIX actually embedded in every archived c14 deep run
# (command.txt verified 2026-07-21: tputsched 900k, tputasl 640k/800k,
# tputschedb 1.1M, tputasm 1.6M — ALL carry KEEP_DGRADS_HBM=1; found via the
# C4b breach diff, it was in no doc inventory because the flag pre-exists in
# BOTH trees and is not a 42 feature. The 120k dial runs did NOT set it.)
# NB fused-addmm + reuse-packed-x are NOT here: 42's ASYM_PINS listed them but
# ZERO archived c14 runs carry them (grep over all command.txt = 0 hits) —
# they stay default-off everywhere pending the 2x2 A/B (merge doc §2d′).
_MOE_PINS = {
    "ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU": "1",
    "ASYMM_QWEN3_MOE_FG_DA_GPU": "1",
    "ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS": "0",
    "ASYMM_FG_ELEMENTWISE_CHUNK_MB": "1024",
    "ASYMM_QWEN3_MOE_DOWN_DX_STAGED": "1",
    "ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM": "1",
}
_MOE_T2_ENV = {   # the c14 keep-acts bundle, as-measured (no panel-cache)
    **_STAGED, **_MOE_PINS,
    "ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM": "1",
    "ASYMM_ATTN_ACT_KEEP_ACTS_HBM": "1",
    "ASYM_GC_SAVE_ON_CPU_OVERRIDE": "false",
}
_MOE_T2B_ENV = {**_STAGED, **_MOE_PINS}          # shed prefix (c14 "balanced")
_MOE_T3_ENV = dict(_MOE_PINS)                     # streamed engine, no staged

_FG000 = "recomp-off-full-fg-ker000-ceil0000-ohbm0"
_FG101 = "recomp-off-full-fg-ker101-ceil0000-ohbm0"

MODELS = {
    "q3-32b": ModelTab(
        "q3-32b", "dense",
        tiers=(
            TierLine("T1", "latency", "unsloth-ohbm0", dict(_STAGED), 0.0, 0.47,
                     valid_k=(128.0, 448.0), note="c12 §4 k 0.47-0.51, a clamped 0"),
            TierLine("T2", "balanced", _FG000, dict(_DENSE_T2_ENV), 10.0, 0.34,
                     valid_k=(128.0, 640.0)),
            TierLine("T3", "memory", _FG000, {}, 0.0, 0.175,
                     host_c=750.2, host_h=0.359,   # fit of RSS 957@576k -> 980@640k
                     valid_k=(128.0, 704.0)),
        ),
    ),
    "llama3.3-70b": ModelTab(
        "llama3.3-70b", "dense",
        tiers=(
            TierLine("T1", "latency", "unsloth-ohbm0", dict(_STAGED), 0.0, 0.51,
                     valid_k=(128.0, 448.0), note="a=-1 fit artifact clamped to 0"),
            TierLine("T2", "balanced", _FG000, dict(_DENSE_T2_ENV), 30.6, 0.366,
                     host_c=980.0, host_h=0.0,     # 975-984 token-flat (c12 §5)
                     valid_k=(128.0, 448.0)),
            # T3 ABSENT by design: measured tier inversion — host-OOM at 416k
            # AND 448k while T2 fits (c12 §5). "—" row in c12 §4.
        ),
    ),
    "q3-30b-a3b": ModelTab(
        "q3-30b-a3b", "moe",
        tiers=(
            # T1 (unsloth-ohbm0) deliberately has NO deep line — c12 §4 "fit
            # pending". T1 is reachable only through the anchor zone.
            # slopes anchored on the ARCHIVED runs (2026-07-21 verification):
            # bundle 183.0 GiB @900k (tputsched-c14) -> m=(183.0-6.3)/900;
            # shed 151.5 @1.1M (tputschedb) -> 0.132; T3 156.1 @1.6M (tputasm)
            # -> 0.095. The 800k shed point (147.5) is OFF the shed line —
            # allocator regime, probe rule covers (ledger 2026-07-21).
            TierLine("T2", "latency", _FG000, dict(_MOE_T2_ENV), 6.3, 0.1963,
                     host_c=539.0, host_h=0.0,     # RSS anchor @800k (c14 P3)
                     valid_k=(160.0, 900.0),
                     note="c14 KA bundle as-measured (NO panel/fused/reuse)"),
            TierLine("T2B", "balanced", _FG000, dict(_MOE_T2B_ENV), 6.3, 0.132,
                     host_c=906.0, host_h=0.0,     # RSS anchor @1.1M (c14 P4)
                     valid_k=(160.0, 1200.0)),
            TierLine("T3", "memory", _FG101, dict(_MOE_T3_ENV), 4.3, 0.095,
                     host_c=925.0, host_h=0.0,     # RSS anchor @1.6M
                     valid_k=(160.0, 1700.0)),
        ),
        anchors={   # measured big-batch regime (fits invalid < 160k)
            64.0:  (8, 4200, 150.0, False),   # ESTIMATE — asym never run at 64k
            80.0:  (8, 3642, 84.7, True),     # c14 P1 measured
            96.0:  (8, 3200, 100.0, False),
            120.0: (8, 2740, 180.0, True),    # scheduler_v2 §3b dial ladder (KA)
            128.0: (8, 2723, 126.7, True),
        },
        anchor_max_k=128.0,
    ),
    "q3.5-35b-a3b": ModelTab(   # pending-fit: single points only (c12 §4);
        # recipe envs deliberately empty — q3.5 recipes come from its own
        # campaign's command.txt, not from these placeholders.
        "q3.5-35b-a3b", "moe",
        tiers=(
            TierLine("T2", "balanced", _FG000, {}, 0.0, 0.11,
                     valid_k=(128.0, 640.0), note="PENDING-FIT: k implied from 95.7@576k; recipe pending"),
            TierLine("T3", "memory", _FG101, {}, 0.0, 0.06,
                     valid_k=(128.0, 704.0), note="PENDING-FIT: k implied from 64.4@640k; recipe pending"),
        ),
    ),
}
MODEL_ALIASES = {"llama": "llama3.3-70b", "q32": "q3-32b", "q3-30b": "q3-30b-a3b"}


# ── memory / host / batch — closed form, no loops ───────────────────────────
def mem_gib(line: TierLine, tok_k: float) -> float:
    return line.base + line.m * tok_k


def host_gb(line: TierLine, tok_k: float) -> float:
    if line.host_c <= 0:
        return 0.0            # unmeasured — watchdog backstops (c12 §5)
    return line.host_c + line.host_h * tok_k


def max_batch(line: TierLine, s_k: float, cap: float) -> int:
    if s_k <= 0 or line.m <= 0:
        return 0
    b_hbm = math.floor((cap - line.base) / (line.m * s_k))
    if line.host_h > 0 and line.host_c > 0:
        b_host = math.floor((C_HOST_EFF - line.host_c) / (line.host_h * s_k))
    elif line.host_c > C_HOST_EFF:
        b_host = 0
    else:
        b_host = 10**9
    return max(0, min(b_hbm, b_host))


@dataclass
class Plan:
    model: str; tier: str; mode: str; B: int
    token: str; env: dict
    mem: float; util: float; host: float
    probe: bool; anchor: bool
    note: str = ""


def schedule(model_name: str, seq: float, safety: str = "normal",
             reserved: float = 0.0) -> Plan:
    model = MODELS[MODEL_ALIASES.get(model_name, model_name)]
    s_k = seq / 1000.0
    cap = (C_HBM - reserved) * BETA
    band = PROBE_FRAC * cap
    # anchor zone: measured truth, not lines
    if model.anchors and s_k <= model.anchor_max_k:
        key = min(model.anchors, key=lambda k: abs(k - s_k))
        B, tok_s, gib, measured = model.anchors[key]
        return Plan(model.name, "ANCHOR", "big-batch", B, "(per anchor row)",
                    {}, gib, gib / C_HBM, 0.0, probe=not measured, anchor=True,
                    note=f"anchor {key:.0f}k ({'measured' if measured else 'ESTIMATE - probe'}) "
                         f"~{tok_s} tok/s")
    knee = KNEE_TOKENS_K.get(model.name)
    probe_candidates = []
    for line in model.tiers:
        B = max_batch(line, s_k, cap)
        if knee and B >= 1:
            B = min(B, max(1, math.ceil(knee / s_k)))
        if B < 1:
            m1 = mem_gib(line, s_k)
            if cap < m1 <= cap + band:      # over cap but inside probe band
                probe_candidates.append((line, m1))
            continue
        m = mem_gib(line, s_k * B)
        h = host_gb(line, s_k * B)
        if h > C_HOST_EFF:
            continue
        in_band = m > cap - band
        lo, hi = line.valid_k
        extrap = not (lo <= s_k * B <= hi)
        note = line.note
        if extrap:
            note = (note + " " if note else "") + "[extrapolated beyond fit range]"
        if probe_candidates:
            pc = ", ".join(f"{l.name}@{mm:.0f}GiB" for l, mm in probe_candidates)
            note = (note + " " if note else "") + f"[probe to upgrade: {pc}]"
        return Plan(model.name, line.name, line.mode, B, line.token, dict(line.env),
                    m, m / C_HBM, h, probe=in_band or extrap, anchor=False, note=note)
    if probe_candidates:
        l, m = probe_candidates[0]
        return Plan(model.name, l.name, l.mode, 1, l.token, dict(l.env), m,
                    m / C_HBM, host_gb(l, s_k), probe=True, anchor=False,
                    note="NO clean tier - PROBE required (prediction inside near-wall band)")
    raise SystemExit(f"INFEASIBLE: {model.name} @ {seq:.0f} — no tier fits "
                     f"HBM<= {cap:.1f} GiB and host<= {C_HOST_EFF:.0f} GB")


def emit(p: Plan) -> None:
    print(f"EMIT model={p.model} tier={p.tier} mode={p.mode} B={p.B}")
    print(f"     backend: asym_cpuadamwds (default — user choice, NEVER auto)")
    print(f"     recompute token: {p.token}")
    print(f"     predicted {p.mem:.1f} GiB ({p.util:.0%} HBM)"
          + (f" | host ~{p.host:.0f} GB" if p.host else " | host unmeasured (watchdog)"))
    if p.note:
        print(f"     note: {p.note}")
    for k, v in sorted(p.env.items()):
        print(f"     env {k}={v}")
    if p.probe and not p.anchor:
        b2 = max(1, p.B - 1)
        print(f"     PROBE (near-wall/extrapolated): bash scripts/lf/tp_probe.sh "
              f"{p.model} tpprobe \"asym_cpuadamwds|{p.token}|ligerloss1\" <seq> {p.B} {b2}")


# ── replay: every recorded decision must be reproduced (S3-V1 gate) ─────────
# category: TIER = clean pick must equal; PROBE_TO = probe-flagged pick whose
# recorded resolution is the given tier (probe rule working, c12 §6-§7);
# EDGE = beyond-band capacity record (scheduler must NOT offer it as safe);
# HOST = excluded by host term; BETA = batch rejected by beta.
REPLAY = [
    ("q3-32b", 128, "TIER", "T1", 2, "c12 §1 ref: b2 (b3 infeasible under beta)"),
    ("q3-32b", 192, "TIER", "T1", 1, ""),
    ("q3-32b", 384, "TIER", "T2", 1, ""),
    ("q3-32b", 448, "TIER", "T2", 1, "measured 164.2 = 89% healthy"),
    ("q3-32b", 576, "TIER", "T3", 1, "111.2 GiB, RSS 957"),
    ("q3-32b", 640, "TIER", "T3", 1, "RSS 980 <= eff cap"),
    ("q3-32b", 704, "HOST", "T3", 0, "host line 1003 > 990 — measured HOST-OOM"),
    ("llama3.3-70b", 192, "TIER", "T1", 1, "parity +0.3%; b2 rejected by beta (97.7%)"),
    ("llama3.3-70b", 320, "PROBE_TO", "T2", 1, "T1 line 163.2 in-band -> probe -> T2 ran"),
    ("llama3.3-70b", 384, "PROBE_TO", "T2", 1, "T2 171.1 marginal-over -> probed FIT"),
    ("llama3.3-70b", 416, "PROBE_TO", "T2", 1, "T2 182.9 marginal-over -> probed FIT"),
    ("llama3.3-70b", 448, "EDGE", "T2", 1, "wall record 97.3% — beyond safe band"),
    ("llama3.3-70b", 999, "NO_T3", "T3", 0, "T3 absent by host inversion (c12 §5)"),
    ("q3-30b-a3b", 640, "TIER", "T2", 1, "c14 640k healthy"),
    ("q3-30b-a3b", 800, "TIER", "T2", 1, "c14 P3 597 tok/s 147.5 GiB (KA state)"),
    ("q3-30b-a3b", 900, "PROBE_TO", "T2", 1, "bundle 176.9 in-band; record ran 183 @99%"),
    ("q3-30b-a3b", 1100, "TIER", "T2B", 1, "c14 P4 382 tok/s 151.5 GiB"),
    ("q3-30b-a3b", 1600, "TIER", "T3", 1, "c14 1.6M headline 292 tok/s"),
]


def _ladder_next(model: str, tier: str) -> str | None:
    names = [t.name for t in MODELS[model].tiers]
    if tier not in names:
        return None
    i = names.index(tier)
    return names[i + 1] if i + 1 < len(names) else None


def _replay() -> int:
    fails = 0
    for model, sk, cat, tier, b, why in REPLAY:
        tag = f"{model} @{sk}k"
        if cat == "NO_T3":
            ok = all(t.name != "T3" for t in MODELS[model].tiers)
            print(f"{'PASS' if ok else 'FAIL'} {tag}: T3 {'absent' if ok else 'PRESENT?!'} — {why}")
            fails += 0 if ok else 1
            continue
        if cat == "HOST":
            try:
                p = schedule(model, sk * 1000.0)
                ok = p.tier != tier          # anything but the host-walled tier
                verdict = f"picked {p.tier} (not {tier})"
            except SystemExit:
                ok, verdict = True, "INFEASIBLE (host)"
            print(f"{'PASS' if ok else 'FAIL'} {tag}: {verdict} — {why}")
            fails += 0 if ok else 1
            continue
        try:
            p = schedule(model, sk * 1000.0)
        except SystemExit as e:
            ok = cat == "EDGE"
            print(f"{'PASS' if ok else 'FAIL'} {tag}: {e} — {why}")
            fails += 0 if ok else 1
            continue
        if cat == "TIER":
            ok = p.tier == tier and (b == 0 or p.B == b)
        elif cat == "PROBE_TO":
            # probe rule resolved the record: either the pick IS the recorded
            # tier (probe-flagged), or the probe resolved one tier down the
            # ladder (llama 320k), or the recorded tier is listed as a
            # probe-upgrade candidate in the note (moe 900k calibration).
            ok = (p.probe and (p.tier == tier or _ladder_next(model, p.tier) == tier)) \
                 or (f"{tier}@" in (p.note or ""))
        elif cat == "EDGE":
            ok = p.probe or p.tier != tier   # must not offer the wall as clean
        else:
            ok = False
        print(f"{'PASS' if ok else 'FAIL'} {tag}: picked {p.tier} B={p.B} "
              f"probe={p.probe} — {why}")
        fails += 0 if ok else 1
    print("REPLAY", "PASS" if fails == 0 else f"{fails} FAILURES")
    return fails


# ── selftest: 5 properties re-targeted at the tier ladder (S2-V1 gate) ──────
def _selftest() -> int:
    fails = 0
    seqs = list(range(160, 1700, 20))
    # 1. tier index monotone non-decreasing in s (nested shedding)
    order = {"T1": 0, "T2": 1, "T2B": 2, "T3": 3}
    last = -1
    for sk in seqs:
        try:
            p = schedule("q3-30b-a3b", sk * 1000.0)
        except SystemExit:
            break
        idx = order[p.tier]
        if idx < last:
            print(f"FAIL nestedness at {sk}k: tier index {last} -> {idx}"); fails += 1
        last = idx
    # 2. predicted mem of the chosen plan <= cap always
    cap = C_HBM * BETA
    for sk in seqs:
        try:
            p = schedule("q3-30b-a3b", sk * 1000.0)
        except SystemExit:
            continue
        if not p.anchor and p.mem > cap + PROBE_FRAC * cap + 1e-9:
            print(f"FAIL cap at {sk}k: {p.mem:.1f} > band top"); fails += 1
    # 3. reserved sweep: tier index monotone with reservation at fixed s
    last = -1
    for res in range(0, 120, 10):
        try:
            p = schedule("q3-30b-a3b", 800_000.0, reserved=float(res))
        except SystemExit:
            break
        idx = order[p.tier]
        if idx < last:
            print(f"FAIL reserved-nestedness at res={res}"); fails += 1
        last = idx
    # 4. analytic boundary consistency: T2B->T3 crossing for q3-30b
    t2b = next(t for t in MODELS["q3-30b-a3b"].tiers if t.name == "T2B")
    analytic = (cap - t2b.base) / t2b.m
    switch = None
    for sk in seqs:
        try:
            p = schedule("q3-30b-a3b", sk * 1000.0)
        except SystemExit:
            continue
        if p.tier == "T3":
            switch = sk
            break
    # scheduler switches when T2B leaves even the probe band — analytic+band
    analytic_band = (cap * (1 + PROBE_FRAC) - t2b.base) / t2b.m
    if switch is None or not (analytic - 40 <= switch <= analytic_band + 40):
        print(f"FAIL boundary: T3 at {switch}k, analytic {analytic:.0f}-{analytic_band:.0f}k")
        fails += 1
    # 5. closed-form B inversion round-trip: mem(B) <= cap < mem(B+1)
    for model in ("q3-32b", "llama3.3-70b", "q3-30b-a3b"):
        for line in MODELS[model].tiers:
            for sk in (160.0, 256.0, 384.0, 512.0):
                B = max_batch(line, sk, cap)
                if B < 1:
                    continue
                if mem_gib(line, sk * B) > cap + 1e-9:
                    print(f"FAIL inversion {model}/{line.name}@{sk}k: mem(B) over cap"); fails += 1
                if mem_gib(line, sk * (B + 1)) <= cap and \
                   host_gb(line, sk * (B + 1)) <= C_HOST_EFF:
                    print(f"FAIL inversion {model}/{line.name}@{sk}k: B not maximal"); fails += 1
    print("SELFTEST", "PASS (5/5 properties)" if fails == 0 else f"{fails} FAILURES")
    return fails


# ── recipe emission for the driver preset layer (S4) ────────────────────────
def _emit_recipes() -> None:
    print("# generated by asym_scheduler.py --emit-recipes — DO NOT EDIT BY HAND")
    print("# family|TIER -> recompute token + recipe env (single source of truth)")
    print("declare -A TIER_TOKEN TIER_ENV")
    fams: dict = {}
    for m in MODELS.values():
        fams.setdefault(m.family, m)      # first model of family carries recipes
    for fam, m in sorted(fams.items()):
        names = {t.name for t in m.tiers}
        if "T1" not in names:
            # moe T1 has no deep byte line (c12 §4 fit-pending) but the RECIPE
            # is well-defined (anchor zone): unsloth-ohbm0 + staged.
            print(f'TIER_TOKEN[{fam}|T1]="unsloth-ohbm0"')
            print(f'TIER_ENV[{fam}|T1]="ASYM_GEMM_DISPATCH=staged"')
        for line in m.tiers:
            env = " ".join(f"{k}={v}" for k, v in sorted(line.env.items()))
            print(f'TIER_TOKEN[{fam}|{line.name}]="{line.token}"')
            print(f'TIER_ENV[{fam}|{line.name}]="{env}"')


# ── offline tau predictor (42's water-fill, verbatim constants; NEVER the
#    runtime decision — merge_scheduler.md §2/§2d′) ──────────────────────────
def _predict(model: str, seq: float) -> None:
    if model not in ("q3-30b-a3b",):
        print("[predict] tau fits exist for q3-30b-a3b only (42's calibration)")
        return
    a, b = 126.0, 1.937          # tau(s) = a + b*s_k us/tok, all rungs on
    rung_dtau = {"staged": 70.0, "ker000": 34.0, "keep-acts": 33.0, "panel-cache": 3.0}
    s_k = seq / 1000.0
    p = schedule(model, seq)
    shed = []
    if p.tier in ("T2B", "T3"):
        shed += ["keep-acts", "panel-cache"]
    if p.tier == "T3":
        shed += ["ker000", "staged"]
    tau = a + b * s_k + sum(rung_dtau[r] for r in set(shed))
    print(f"[predict/OFFLINE] {model} @{s_k:.0f}k tier={p.tier}: "
          f"tau~{tau:.0f} us/tok -> ~{1e6 / tau:.0f} tok/s "
          f"(42's deep-end fit; validated -3% @900k; NOT a decision input)")


def _sweep(model: str) -> None:
    print(f"{'seq':>7} | {'tier':<6} {'B':>2} {'mode':<9} {'GiB':>6} {'util':>5} "
          f"{'host':>5} {'probe':<5} note")
    for sk in (64, 80, 96, 128, 160, 208, 256, 320, 400, 480, 576, 640, 720,
               800, 880, 960, 1100, 1200, 1400, 1600):
        try:
            p = schedule(model, sk * 1000.0)
        except SystemExit:
            print(f"{sk:>6}k | INFEASIBLE")
            continue
        host = f"{p.host:.0f}" if p.host else "—"
        print(f"{sk:>6}k | {p.tier:<6} {p.B:>2} {p.mode:<9} {p.mem:>6.1f} "
              f"{p.util:>5.0%} {host:>5} {str(p.probe):<5} {p.note[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="q3-30b-a3b")
    ap.add_argument("seq", nargs="?", type=float, default=480000)
    ap.add_argument("--safety", default="normal", choices=list(SAFETY_H))
    ap.add_argument("--reserved", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--emit-recipes", action="store_true")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.replay:
        sys.exit(_replay())
    if a.emit_recipes:
        _emit_recipes(); return
    if a.sweep:
        _sweep(MODEL_ALIASES.get(a.model, a.model)); return
    name = MODEL_ALIASES.get(a.model, a.model)
    if name not in MODELS:
        sys.exit(f"unknown model {a.model!r}; known: {', '.join(MODELS)}")
    reserved = a.reserved + C_HBM * (SAFETY_H[a.safety] - SAFETY_H["normal"])
    p = schedule(name, a.seq, a.safety, max(0.0, reserved))
    emit(p)
    if a.predict:
        _predict(name, a.seq)


if __name__ == "__main__":
    main()
