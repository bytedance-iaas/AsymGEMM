# system_summary — the complete scheduler/system design (self-contained; 2026-07-20)

## §0 ABSTRACT (the whole design in one paragraph)

The system trains LoRA-SFT with frozen base weights resident in HOST memory and
offers a small set of discrete configurations ("tiers") that trade GPU memory for
speed. The scheduler is: **fitted straight lines per configuration** — a memory line
`M = base + m·(B·s)` [GiB, linear in tokens], a speed line `τ = a + b·s` [us/token],
and a host-RAM screen — plus **one argmax**: keep every configuration whose memory
lines fit under the GPU and host capacities, predict tok/s = 10⁶/τ(s), emit the
fastest (ties within run noise → least memory). Batch is DERIVED
(`B* = min(⌈N*/s⌉, largest-B-that-fits)`), never entered. "Latency / balanced /
memory mode" and the superoffload fallback are OUTPUTS of the argmax, not settings;
every mode boundary is a closed-form crossing of a memory line with device capacity.
Goal achieved and measured: parity-or-better vs superoffload wherever it exists,
+6–37% at long seqs, sole coverage far beyond every baseline's wall (q3-30b: 1.6M
tokens = 2.4× the best baseline; q3.5-122b: crossovers arrive ~10× earlier, bigger wins).

---

## §1 THE CONFIG SPACE — backends and the three asym tiers

Baselines (DeepSpeed superoffload; weights streamed by ZeRO machinery, stock HF
compute): **so-recomp** (plain gradient-checkpointing) · **so-unsloth** (unsloth GC,
recompute-saves stay in GPU) · **so-unsloth-OFF** (saves offloaded to CPU; leanest
HBM, heaviest host).

Asym backend (ours; weights PINNED in host RAM, our fine-grained LoRA wrapper +
CPU-AdamW) has 3 optimization sets, shed in fixed order (C→B→A) as memory tightens:

| set | flags | speed when OFF | memory saved when OFF |
|---|---|---|---|
| A engine | `ASYM_GEMM_DISPATCH=staged` + ker000 (vs asym-kernel + fused ker101) | +104 us/tok | ~2 GiB + 0.04 GiB/1k tok |
| B keep-acts | `ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1` + `ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1` + `ASYM_GC_SAVE_ON_CPU_OVERRIDE=false` | +33 us/tok | ~0.05 GiB/1k tok (≈50 GiB @1M) |
| C panel cache | `ASYM_W_PANEL_CACHE_GB=6` | +3 us/tok (sub-noise; kept — free) | 6 GiB |

**T1 LATENCY = A+B+C · T2 BALANCED = A only · T3 MEMORY = none.** Shed order =
ascending value density (tok/s bought per GiB occupied: C 0.5 < B 0.7 < A 2.9) — a
knapsack eviction computed from measured prices, not hand-chosen. A is one coupled
set (fused ker101 exists only inside the slow engine; staged+ker101 impossible,
asym+ker000 dominated). Class-1 pins (always on, never dialed): chunking 1024,
staged down-dx, GPU LoRA-A/dA, fused-addmm, packed-X reuse, CPU AdamW, liger loss.
Hybrid-attention models (qwen3.5: 3/4 of layers linear-attention): set B currently
covers the MoE + full-attention layers only — the linear-attention module's port is
pending (remaining_optimizations.md item 7b); until then B's measured prices on those
models are partial (which is why 122b asym still moves ~1.1 TB/step of copies).

## §2 THE SCHEDULER — two lines + one argmax (the paper formulation)

```
Given: model (arch), seq s, device (C GiB HBM, D GiB host), safety h ∈ {2,5,8}%
Per config x (each baseline + each asym tier), fitted from ≥2 past runs:
    MEMORY LINE   M_x(B,s) = base_x + m_x·(B·s)      [GiB; machine-independent]
    SPEED LINE    τ_x(s)   = a_x + b_x·s             [us/token; per hardware class]
    HOST LINE     D_x(B,s) = wbase_x + d_x·(B·s)     [GB; wbase = pinned/streamed weights]
Derived batch:  B*_x = min( ⌈N*/s⌉ , ⌊(C·(1−h) − base_x)/(m_x·s)⌋ )
                 (N* = knee ≈ 0.4M tokens for q3-30b: tok/s is batch-flat above — measured b2=b4=b5)
Feasible:       M_x ≤ C·(1−h)  AND  D_x ≤ D_wall (~957 GB on GB200 nodes)
Edge penalty:   ×(1+ε), ε = 0 below 92% util, 0–6% at 92–98% (measured), ∞ above 98%
Throughput:     TP_x = 10⁶/τ_x(s) · min(1, B·s/N*) · 1/(1+ε)
                (below the knee, tok/s scales ≈ linearly with tokens/step; above, flat)
EMIT:           argmax over feasible x of TP_x; ties within run noise (±1.5%) → min memory
```
Host-line honesty: D_x is the coarsest of the three fits — CPU pools reuse and
plateau (measured RSS 537→539 across 640k→800k, then 906→925 at 1.1M→1.6M), so
treat the D line as an upper-bound screen and the measured host walls (e.g.
uns-OFF ≈1.05M tokens; asym T3 925/957 @1.6M) as the real gate.

Properties (machine-checked, `scripts/lf/asym_scheduler.py --selftest`, 5/5): configs
are NESTED along seq (flags only shed, in fixed reverse-value order, never reshuffle);
tok/s monotone in s; same nesting under reserved-memory cuts and the safety dial.

**Hardware-agnosticity:** tier/mode selection never touches timing — boundaries are
byte-threshold crossings `s* = (C(1−h) − base − Σdm_fixed)/(m + Σdm_tok)`, with C a
device spec and slopes derivable from model architecture (proven machine-independent:
c12-vs-c14 peak-reserved bit-identical). Timing (the a,b fits) affects ONLY the
sup-vs-asym choice inside the overlap window — worth ±2–4% — and only via the SIGN
of a_sup − a_asym; a 2-probe calibration per hardware class refines it. Degenerate
safe policy: always emit asym → lose ≤4% in the fallback window, all capacity/mode
decisions unchanged.

**Maintenance loop:** every emission carries predicted (tok/s, GiB); a probe
disagreeing beyond ~10% refits that one config's line (2 runs). Two real examples:
the 900k emission measured −3% (accepted); keep-acts memory turned superlinear
>800k → slope refit moved the T1→T2 boundary 987k→~894k automatically.

## §3 FITTED CONSTANTS (q3-30b-a3b, GB200-185, b1 deep-end; s_k = seq in 1000 tokens)

| config | τ(s) us/tok | M(s,1) GiB | b1 wall predicted → measured |
|---|---|---|---|
| asym T1 (A+B+C) | 126 + 1.94·s_k | 4.3 + 0.179·s_k (keep-acts superlinear >800k: 0.052/ktok refit) | 987k → 900k ran @99%; T1→T2 ~894k |
| asym T2 (A only) | rung-sum 159 + 1.94·s_k; 1.1M measured implies ~480 intercept (−12% tok/s vs pred — single-point refit pending a 2nd anchor) | ~6.3 + 0.138·s_k (T1 minus keep-acts dm) | ~1.25M (T2→T3 crossing; 1.1M ran @82% ✓) |
| asym T3 (none) | ~300 + 1.94·s_k (rung-sum predicts 266 = T1+3+33+34+70; fitted 326 from the clean 1.6M point — slow-engine interaction ≈+13%; the 1.4M point reads high, 562-implied, under near-wall host pressure RSS 940) | ~5 + 0.119·s_k | ~1.5M HBM but HOST-bound → 1.6M ran @84%, RSS 925/957 ✓ |
| sup-unsloth | −63 + 2.24·s_k | 19.9 + 0.253·s_k | 638k → **(640k, 660k] ✓** |
| sup-recomp | 62 + 2.00·s_k | 4.1 + 0.452·s_k | 391k → **(392k, 400k] ✓** |
| sup-unsloth-OFF | single deep point (1808 us/tok @800k); short-seq b8 regime separate | HBM-lean (118 @800k) | HOST wall ≈1.05M tokens (800k-tok fit / 1.1M-tok watchdog-OOM, measured) |

Knee N* ≈ 0.4M tokens · edge penalty 0–6% (92–98%) · near-wall allocator bend −3…−5
GiB (predictions within 5 GiB of capacity are coin-flips — probe them) · run noise
~1.5% (parity gate ±2%, beat gate >+5%).
PIECEWISE CAVEAT: the lines above are the b1/b2 DEEP-END fits. The big-batch
short-seq regime (below the knee) has different effective intercepts (MFU regime,
c_fix amortization) — the scheduler uses measured short-seq anchor points there
(e.g. 80k/128k rows in §5) rather than extrapolating the deep-end lines backwards.
Two regimes, one rule: anchors where measured, lines where fitted, never cross-extrapolate.

## §4 EMITTED BEHAVIOR (q3-30b; every boundary from §2's lines, none hand-set)

| seq | emission | measured result |
|---|---|---|
| ≲130k | asym T1, batch to the knee | **+6…23% over baselines** at half the HBM (80k: 3642 vs 3424) |
| 160–600k | **sup-unsloth fallback** (its a is lower) | parity by construction (asym −1.5…−4% if forced) |
| 600–660k | asym T1 (tie→leaner) | 640k: 732 vs 731 = parity at sup's last fit, 60% vs 98% HBM |
| 660k–894k | asym T1 | sole coverage (800k: 597, +8% over uns-OFF 553; 900k: 519 @99%) |
| ~894k–1.25M | asym T2 (keep-acts shed) | 1.1M: 382 — all baselines dead (uns-OFF host-OOM proven) |
| >1.25M | asym T3 (engine shed) | 1.4M: 305 @73% · **1.6M: 292 @84%, RSS 925 — max-seq headline** |

## §5 CROSSOVER EVIDENCE (the five measured points, q3-30b; cell = lat·TP·HBM(%)·RSS)

| seq | so-recomp | so-unsloth | so-unsloth-OFF | asym (tier) | verdict |
|---|---|---|---|---|---|
| 80k b8 | ~4400 est | 186.9·3424·176.9(96%)·364 | 228.1·2806·94.4·599 | T1: 175.7·**3642**·84.7(46%) | P1 ✅ +6%, half the HBM |
| 640k b1 | OOM | 875.4·731·181.5(98% edge)·382 | fits | T1: 873.8·**732**·111.4(60%)·537 | P2 ✅ parity at sup's edge |
| 800k b1 | OOM | OOM | 1446.8·553·118.1(64%)·663 | T1: 1340.0·**597**·147.5(80%)·539 | P3 ✅ +8% over last-alive |
| 1.1M b1 | OOM | OOM | **HOST-OOM** (watchdog) | T2: 2879.3·**382**·151.5(82%)·906 | P4 ✅ all baselines fail |
| 1.4M b1 | OOM | OOM | HOST-OOM | T3: 4589.9·**305**·134.2(73%)·940 | P5 ✅ T3 alone |
| 1.6M b1 | OOM | OOM | HOST-OOM | T3: 5487.6·**292**·156.1(84%)·925 | max-seq = **1.6M** (2.4× best baseline) |

q3.5-122b (in progress, same pattern ~10× earlier): @32k×b8 rc=GPU-OOM,
uns-OFF=host-OOM, uns=909, asym T1=841 (−7.5%, −40 GiB); @b1 deep-end uns edge-locks
from 288k (665@97%, 640@98% @320k) while asym T1 @384k = **874 (+37%) at 85%** —
hybrid linear attention flattens asym's τ (b_x ≈ 0) while uns's grows steeply.

## §6 HONEST-DEFENSIBILITY NOTES (what to claim, what to downplay)

1. **asym T1 ≡ superoffload mechanism class** (host weights + on-demand staging +
   fast GEMMs + saves-on-GPU + CPU optimizer): different residency↔traffic operating
   point — T1 holds ~30% less per token and moves more (capacity edge and small
   speed tax are the same coin). Downplay T1-vs-sup capacity (~15 GiB is DeepSpeed
   working-set overhead, erodable); the leaner-saved-set design (finer-grained
   recompute, active in ALL tiers) is a real but same-class choice.
2. **The defensible novelty:** (a) T3's stage-free asym GEMM (dense GEMM computed
   directly against CPU-resident weights over C2C — the capacity frontier no staged
   system reaches); (b) THIS scheduler formulation (fitted-line cost models, one
   argmax, modes/batch/fallback as outputs, machine-checked monotonicity, predict→
   probe→refit); (c) the measured frontier map (walls at ≤20k granularity, knee law,
   edge regimes, parity crossover, sole-coverage window).
3. **The fallback is an output**: sup is emitted where its τ intercept wins; every
   asym per-token fix moved that window automatically (it shrank 640k→600k-ish over
   the five fix rungs — the design's self-updating property, demonstrated).

## §7 OPERATIONS (how to run it)

`python scripts/lf/asym_scheduler.py MODEL SEQ [--safety normal] [--reserved GiB]`
→ prints family, B*, flag set, predicted tok/s + GiB (the contract to verify);
`--sweep` regenerates the §4 table; `--selftest` re-checks the 5 properties.
Protocol for any probe/refit run: `PROFILERS=source WARMUP_STEPS=1 MAX_STEPS=2`
(steady = mean of the 2 measured; ~1% step variance), serial GPU, guard = GPU-empty
AND host-avail >1.5T before launch, kills by PID only (never pkill), archive to
profiling_tp_$(hostname)/. New-model onboarding = cluster of 4 runs at one short seq
(all configs) + 2 deep-end b1 probes per surviving config → fit lines → sweep.
