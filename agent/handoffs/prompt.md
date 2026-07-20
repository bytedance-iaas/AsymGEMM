# Scheduler interface redesign — from price (β) to budget (M*) · 2026-07-19

The core-system design record. Question answered: "β = tok/s-per-GiB is too
hardware-dependent and uninterpretable — is there a more tractable, interpretable,
comprehensive scheduler formulation?" Verdict: YES — same math, different exposed
variable. The user-facing knob must be the MEMORY BUDGET (dual of β), plus one
3-position safety dial. β survives only as internal machinery (the fixed knob order).

================================================================================
## 1. Why β fails as an interface (the critique, accepted)
================================================================================
1. Uncalibratable: no user can pick "2 vs 50 tok/s per GiB" — the number has no
   referent in their world. GiB does (nvidia-smi shows it).
2. Hardware/model-dependent: the meaningful β range shifts whenever the knob prices
   (ρ) shift — every new GPU or model invalidates learned intuition.
3. Unverifiable: a user can check "peak memory ≤ budget I set"; they cannot check
   "the system honored my price."
4. Wrong question: for a single job on a dedicated GPU there is NO genuine
   latency↔memory tradeoff to express — if it fits with everything on, the fastest
   config wins, full stop. Memory only has value when something ELSE wants it
   (co-tenant, a serving job's KV cache, safety margin, a second experiment).
   All of those are naturally expressed as a BUDGET, never as a price.

================================================================================
## 2. The redesigned interface (complete: two knobs, both physical, both optional)
================================================================================
```text
schedule(model, seq_len, hbm_budget = DEVICE_FULL, safety = normal)

hbm_budget : GiB (or % of device). "This job may use at most this much GPU memory."
             Default = whole device ⇒ pure speed-first behavior.
             THIS is the latency↔memory steering wheel:
               budget 100% → most-latency-focused the workload allows
               budget  55% → memory-focused (co-tenant friendly, deep safety)
             Monotone + nested: lowering the budget only shrinks the config,
             in one fixed measured order — no oscillation ever.
safety     : {conservative, normal, aggressive} = how close to the budget edge the
             system may sail = headroom h ∈ {8%, 5%, 2%}. Aggressive accepts the
             measured 0–6% near-edge slowdown and coin-flip walls; conservative
             never exceeds 92% of budget. 3 positions suffice (measured penalty
             structure has exactly 3 regimes: free / edge / infeasible).
```
The user NEVER chooses modes, batches, knobs, or prices. "latency mode" /
"memory mode" become LABELS ON THE OUTPUT (reports), not inputs.

Batch is an output: B* = min(B_knee(s), B_max(s; budget)) — the smallest batch
that saturates the GPU, capped by capacity. (Knee N*≈0.4M tokens for q3-30b:
below it batch buys tok/s; above it tok/s is batch-flat — measured b2=b4=b5
@208k — so extra batch is pure memory waste and is not taken.)

================================================================================
## 3. The solver (identical machinery as scheduler_v2 §4/§8/§9 — budget-form)
================================================================================
```text
C_eff = min(hbm_budget, C_phys) · (1 − h)
for each family f ∈ {asym, sup-unsloth, sup-recomp}:            # sup = fallback candidates
    B_f = min(B_knee(s), B_max_f(s; C_eff))                     # batch = output
    x_f = class-1 base of f
    for knob k in FIXED ρ-descending ladder of f:               # staged → ker000 → keep-acts…
        admit k iff M(x_f + k) ≤ C_eff                          # budget check, nothing else
    U_f = predicted tok/s(x_f, B_f)                             # affine fits, scheduler_v2 §9.2
emit argmax_f U_f  (+ its predicted tok/s and peak GiB — the user-visible contract)
probe-vs-predict mismatch > tol ⇒ refit that family's 4 constants (2 probes) — the
system self-calibrates per (model, hardware); users never touch constants.
```
β's role after the redesign: the ρ-DESCENDING LADDER ORDER *is* the price system,
measured once and baked in (staged 9.2 → ker000 0.90 → keep-acts 0.87 s/GiB …).
Budget-greedy walks the same convex frontier as a β-sweep (LP duality): nothing is
lost mathematically; only the exposed variable changes.

================================================================================
## 4. Behavior — the user's stated goal falls out at DEFAULT settings
================================================================================
Goal: "parity on not-so-long seqs; real advantage on very long and ultra-long."
With hbm_budget = full device, safety = normal (q3-30b, GB200-185, measured):

```text
seq        budget state                 emitted                     result vs baselines
≤130k      slack (all knobs fit)        asym full-latency stack,    +6..23% (WINS; sup's
                                        batch to the knee           thrash rows auto-avoided)
160–600k   slack at b1–b2               argmax = sup-unsloth        PARITY BY CONSTRUCTION
                                        (fallback emerges; asym     (asym within −1.5..−3.9%
                                        −1.5% post-fix if forced)   post-fix)
600–660k   sup at its edge/wall         asym-lat (tie→leaner at     parity → win, 40 GiB slack
                                        640k: 732 vs 731)
660–990k   sup infeasible               asym-lat                    SOLE COVERAGE (800k ✓)
990k–1.5M  asym-lat infeasible          knobs shed in ladder order  SOLE COVERAGE, deeper
                                        → asym-mem emerges          (174k×8 anchor ⇒ ~1.5M b1)
```
The latency→memory "orientation" over seq is NOT chosen by anyone: it is the
shadow of the fit constraint tightening as tokens grow. Lowering hbm_budget shifts
every boundary left by exactly (budget cut)/slope_f — predictable, linear,
verifiable.

================================================================================
## 5. Why this is the right formulation (defense, point by point)
================================================================================
1. Interpretable: both knobs are physical (GiB, % risk). Verifiable post-hoc
   (peak-reserved ≤ budget: yes/no).
2. Hardware-portable: budgets transfer as %; the measured constants recalibrate
   from 2 probes per family (the existing scheduler_v2 §8 learning loop).
3. Clean/monotone: nested-config guarantee inherited unchanged from the ρ-ladder
   (lower budget ⊆ higher budget's config; fixed shed order; no re-ordering —
   scale-invariance of ρ, scheduler_v2 §2.3).
4. Complete: multi-tenancy, safety margin, host pressure (same budget logic on
   host RAM: dram_budget → ohbm dial) all expressible; a cluster-level scheduler
   that DOES want prices can sit above and set per-job budgets (two-level design:
   prices between jobs, budgets within a job).
5. Honest about the only real gap: the dial is a ~4-step staircase per family;
   graded per-layer keep-acts (D3-graded, designed, unbuilt) makes the budget
   response fully continuous. Until built, the system rounds DOWN to the nearest
   rung that fits (never violates budget).

================================================================================
## 6. Migration notes (minimal changes)
================================================================================
- scheduler_v2 §9 stays valid (the math); §9.5's "β tuning recipe" is DEMOTED to
  internal calibration; this file's §2 interface supersedes it as the user contract.
- Emitted-config reports print: family, B*, knob set, predicted tok/s, predicted
  peak GiB, budget-utilization %.
- Profile sugar over the single budget knob: --profile latency == budget 100% ·
  balanced == 80% · memory == 55% (percentages from the measured rung boundaries
  at the 120k×8 anchor: L3=180 GiB, L1=107.6, L0=100.3 of 185).

================================================================================
================================================================================
## v2 (2026-07-19, same day) — THE ALLOCATION FORMULATION (supersedes §2 above)
================================================================================
Second critique, accepted: "nobody limits HBM — we always max it out while keeping
throughput high." Correct ⇒ the budget knob is ALSO wrong as the primary interface
(kept only as optional `reserved_hbm` for multi-tenancy, default 0). The honest
formulation has NO preference knob at all:

### The problem, restated (reviewer-clean)
HBM is a fixed endowment C. Spending it has exactly TWO productive uses, with
measured, workload-dependent marginal returns:
```text
asset 1: BATCH units      cost m_f(s) GiB/sample     return: large below the knee
                                                     (N < N*), ≈ 0 above (measured
                                                     b2=b4=b5 @208k — batch-flat)
asset 2: RESIDENCY knobs  cost ΔM_k(s) ∝ tokens      return: per-token time saved
         (staged panels, route bufs, keep-acts)      (measured ρ ladder)
(everything else is class-1, always on, or headroom h — the risk floor)
```
### The scheduler = one greedy water-fill, one ranking
```text
allocate(model, s):  x ← class-1 base of each family, B ← 1
  repeat: buy the SINGLE next unit — one more sample OR one more knob —
          with the highest marginal Δtok/s per GiB, until C·(1−h) is spent
          or no unit's return > 0;  emit argmax over families.
```
This UNIFIES the old two rules: B* = min(knee, capacity) is now a THEOREM (batch
units price themselves out of the ranking at the knee), and the knob ladder is the
same ranking's tail. Batch and residency COMPETE for the same GiB — which is the
real physics, and is what "accommodate a larger batch instead of thrashing" means
formally: at short seq the allocator outbids residency with batch below the knee,
and it never buys >98% utilization (infeasible) or edge rows that lose tok/s
(the 320k b2-edge 1355 < b1 1436 is auto-rejected by return, not by rule).

### What remains user-facing (nothing preferential)
```text
inputs:   model, seq  (+ hardware, auto-detected)
optional: safety h ∈ {2,5,8}%   — risk, not preference
          reserved_hbm (GiB, default 0) — multi-tenant carve-out, not a dial
outputs:  family + batch + knob set + predicted (tok/s, peak GiB)
          + the LABEL ("latency-mode", "balanced", "memory-mode") as a REPORT
```
"Latency-focused vs memory-focused" is what the emitted portfolio IS at that
(model, seq) — short seqs come out batch+residency-rich (latency-looking), ultra-
long comes out residency-shedding (memory-looking) — steering happens by the
workload, which is exactly the user's original ask ("given model+seq, determine
the orientation"). No human-set tradeoff parameter exists to mis-set.

### Alternatives considered and rejected (review trail)
β price — uninterpretable (v1 §1). Budget — nobody limits HBM; demoted to
reserved_hbm. Pareto-menu ("pick one of 4 configs") — pushes the decision back to
the user; kept as a REPORT only. SLO-form ("hit X tok/s, min memory") — inverse
problem for serving, derivable from the same frontier, out of scope for training.
Allocation/water-fill — ADOPTED: single objective (throughput), fixed endowment,
measured marginal returns, modes emergent, zero preference parameters.

================================================================================
## v2 VALIDATION LOG (2026-07-19) — design test + ultra-long GPU runs
================================================================================
OFFLINE (scripts/lf/asym_scheduler.py --selftest): 5/5 cleanness properties PASS —
nested shedding along seq · shed order = reverse-ρ (panel-cache→keep-acts→ker000→
staged) · tok/s monotone · reserved-hbm nested · safety-dial monotone. Sweep emits:
anchors ≤128k → sup fallback 160–560k → asym-latency 640–880k → balanced ~940k+ →
staged-only ~1300k+ (boundaries recomputed from constants after every refit).
GPU RUN 1 (tputsched-c14, 900k b1, latency emission incl. panel-cache): **FITS at
183.0 GiB (98.9%), 519 tok/s vs 535 predicted (−3.0% ✓)**. Calibration finding:
keep-acts memory is SUPERLINEAR past ~800k (0.295 GiB/1k secant vs 0.179 fit) —
folded into the rung table (dm 0.038→0.052/ktok); lat→balanced boundary moved
987k→~894k; selftest boundary assertion now derives from the constants (refit-proof).
GPU RUN 2 (tputschedb-c14, 1.1M b1, balanced emission): launched, manually
interrupted at step 0 (no crash, no OOM; GPU freed). PENDING — relaunch command:
env per the balanced emission (staged+ker000+pins, NO keep-acts flags, NO GC
override, MAX_SAMPLES=512 MAX_STEPS=3), RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|
recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1 ; 1100000|1|1 ; none|false|
false|false|false|false". Gates: fits (~158–175 GiB est), tok/s ≈ 436 ±10%,
OOM would itself be a valid bracket → allocator sheds ker000 next.

================================================================================
## v3 (2026-07-20) — HARDWARE-AGNOSTICITY CLARIFICATION (user critique, accepted)
================================================================================
Critique: "τ (us/tok) varies across machines — unreliable as a scheduler input;
should the heuristic be threshold-like and hardware-agnostic?" Resolution: the
formulation ALREADY is, once stated in the right order. Decompose the decisions:

1. TIER/MODE SELECTION (T1→T2→T3) = PURE MEMORY THRESHOLDS — no timing anywhere.
   s*_tier = (C·(1−h) − base − Σ dm_fixed) / (m + Σ dm_per_tok)
   Every quantity is bytes: C = device HBM (a spec, not a measurement); the slopes
   are ARCHITECTURE-DERIVED (bytes/token/layer from model config: h, L, experts,
   attention type) — analytically computable before ever running, and machine-
   independent (PROVEN: c12 vs c14 peak-reserved BIT-IDENTICAL on every confirm).
   This is exactly the "thresholding" form: seq crosses a byte line → shed a flag.
2. BATCH = the same byte arithmetic + one model constant (knee N*, a property of
   the MODEL's compute intensity, not of the host).
3. FAMILY CHOICE (asym vs sup fallback in the mid-window) = the ONLY timing-
   dependent decision — and it is (a) worth only ±2–4% post-fixes, (b) dependent
   only on the SIGN of (a_sup − a_asym), i.e. an ORDERING, not an absolute rate;
   orderings are stable within a hardware class even when absolute us/tok shifts.
   Degenerate-but-safe default: skip timing entirely, always emit asym → lose
   ≤4% in the fallback window, keep every capacity/mode decision exact.
4. τ's remaining role = OPTIONAL REFINEMENT: 2 probes/family calibrate the
   fallback boundary and the predicted-tok/s contract on THIS machine. Portability
   rule: express τ ratios (config ÷ sup-baseline at same seq) — dimensionless,
   transfers across machines far better than microseconds.

Paper framing: "mode selection is closed-form byte thresholding from model
architecture + device capacity; a 2-probe timing calibration optionally refines
the baseline-fallback boundary" — hardware-agnostic where it matters, calibrated
only where calibration is cheap and optional.
