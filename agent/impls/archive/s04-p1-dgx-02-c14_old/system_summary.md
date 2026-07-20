# system_summary — the defensible scheduler/mode design (2026-07-19, c14)

Purpose of the system: **match superoffload wherever it exists, beat it on kinda-long
sequences, and keep training at ultra-long sequences where every baseline is OOM-dead.**
The scheduler (scripts/lf/asym_scheduler.py; design record agent/handoffs/prompt.md)
selects between three discrete tiers of the asym backend plus the superoffload
fallback — automatically, from (model, seq): batch, tier, flags are OUTPUTS, never
user inputs. Evidence: c14 campaign tables (test_throughput_results.md), fix ledger
(fix_asym.md), dial ladders (scheduler_v2.md §3b/§7).

## Table 1 — tier boundaries (q3-30b-a3b, GB200 185 GiB, b1, normal safety)

| seq range | tier | why the boundary is there |
|---|---|---|
| < ~900k | T1 LATENCY (all flags on) | keep-acts still fits; 900k measured at 99% HBM = the edge |
| ~900k – ~1.25M | T2 BALANCED (drop keep-acts) | saved activations no longer fit; routing buffers still do |
| > ~1.25M (to ~1.4–1.5M est) | T3 MEMORY (drop ker000, then staged) | routing buffers stop fitting; the stage-free kernel goes deepest |

x ≈ 900k, y ≈ 1.25M. Footnotes: (1) inside the T1 range, the 160–600k stretch emits
the so-unsloth fallback instead — same mechanism class, worth only a few %, not a
tier change; (2) boundaries are DERIVED from fitted constants and shift predictably
with batch (at b2 they roughly halve), model, and safety; the scheduler recomputes
them — this table is what it prints for this hardware+model at b1.

## Table 2 — what each tier is for (goal → precise, measured version)

| tier | intuitive goal | precise version (measured) |
|---|---|---|
| T1 LATENCY | "get on par with so-recomp/unsloth" | Parity is the floor, not the goal. Mid-band (160–600k): parity (−1.5…−4%; fallback covers the sliver). Short (<130k): T1 BEATS the baselines outright (+6–23%, larger batch below the knee). 600–660k: matches them (640k: 732 vs 731 tok/s) while they ride a 98% edge and T1 sits at 60%. 660–900k: SOLE COVERAGE — T1's band already extends past every baseline's wall (900k validated at 99% HBM, 519 tok/s, −3% vs prediction). |
| T2 BALANCED | "beyond baseline capacity but still fast" | Exactly. All of 900k–1.25M is past every baseline's wall; the only competition is our own T3, and T2 keeps the fast GEMMs (~7–8% over what T3 would give there) at moderate memory. (1.1M validation run pending.) |
| T3 MEMORY | "aim only at capacity" | Yes. Nothing else reaches 1.25M+; the stage-free kernel + fused routing is the leanest per token (0.119 GiB/1k-tok class) — pure max-sequence and max-batch-per-GiB (measured 174k at b8 = 170.9 GiB, +118% capacity over unsloth's b8 ceiling). Speed ~20–25% below T2. |

## The tiers, concretely (3 discrete flag sets, shed in fixed value order)

| set | flags | speed price when OFF | memory saved when OFF |
|---|---|---|---|
| A engine bundle | ASYM_GEMM_DISPATCH=staged + ker000 (vs asym kernel + ker101) | ~104 us/tok | ~2 GiB + ~0.04 GiB/1k tok |
| B keep-acts | ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1 + ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1 + ASYM_GC_SAVE_ON_CPU_OVERRIDE=false | ~33 us/tok | ~0.05 GiB/1k tok (≈50 GiB @1M) |
| C panel cache | ASYM_W_PANEL_CACHE_GB=6 | ~3 us/tok (sub-noise; kept: mechanism-verified, free) | 6 GiB flat |

T1 = A+B+C · T2 = A only · T3 = none. Shed order C→B→A = ascending value density
(tok/s bought per GiB occupied: C 0.5, B 0.7, A ~2.9) — a knapsack eviction, computed
from measured prices, not hand-chosen. A is one coupled set: ker101's fused routing
exists only inside the asym engine (staged+ker101 impossible; asym+ker000 dominated).

## Honest-defensibility notes (what to claim, what to downplay)

1. **T1 vs superoffload is the SAME mechanism class** (host-resident weights,
   on-demand staging, fast GPU GEMMs, saves mostly on GPU, CPU optimizer). They sit
   at different residency↔traffic balances: sup holds more resident and moves less
   (hair faster); T1 holds ~30% less per token and moves more (measured memcpy 15–18
   vs 4.5 us/tok @208k) — the capacity edge and the speed tax are the same coin.
2. **Downplay T1's capacity edge over sup** (900k vs 660k wall): ~15 GiB of it is
   DeepSpeed working-set overhead (pure implementation disparity, erodable by a
   patch); the ~30%-leaner per-token slope is a real design choice (finer-grained
   recompute saves a smaller set — recomp-off-full-fg is active in ALL tiers; B only
   moves WHERE the saved set lives, not WHAT is saved) but still same-class. Frame as
   "a leaner implementation of the same mechanism, incidentally +36% coverage."
3. **The defensible novelty**: (a) T3's stage-free asym GEMM primitive (dense GEMM
   computed directly against CPU-resident weights over C2C — the capacity frontier no
   staged system reaches); (b) the scheduler formulation (measured affine cost model,
   one knapsack over batch+residency, modes as emergent labels, baseline fallback by
   argmax, predict→probe→refit; machine-checked nested/monotone behavior — 5/5
   selftest); (c) the measured frontier map (per-config walls at ≤20k granularity,
   batch-flat knee law, edge-penalty regimes, the 640k parity crossover, sole
   coverage 660k→900k+ validated on GPU).
4. **The fallback is an output, not a rule**: sup is emitted in 160–600k because its
   predicted tok/s wins there — every asym per-token improvement moves that window's
   left edge automatically (it moved 640k→600k-ish across the five fix rungs).

## Result tables — one per model (cell = tok/s · resv GiB (%HBM) · batch; — = not run)

### q3-30b-a3b (MoE)

| ctx | asym T1-LATENCY | asym T2-BALANCED | asym T3-MEMORY | so-unsloth | so-recomp |
|---|---|---|---|---|---|
| 80k | 3642 · 84.7 (46%) · b8 | — | ~2300 · 80.1 (43%) · b8 | 3424 · 176.9 (96%) · b8 | — |
| 120k | 2723 · 126.7 (68%) · b8 | 2963 · 107.6 (58%) · b8 | 2124 · 100.3 (54%) · b8 | 2212 · 150.1 (81%) · b8 (so-off) | — |
| 128k | — | — | — | 3055 · 149.9 (81%) · b4 | 2985 · 119.5 (65%) · b2 |
| 174k | — | — | 878 s/it · 170.9 (92%) · b8 | — | — |
| 208k | 1883 · 78.1 (42%) · b2 | — | — | 2099 · 122.5 (66%) · b2 | 2056 · 98.7 (53%) · b1 |
| 320k | 1336 · 117.5 (63%) · b2 | — | — | 1436 · 94.8 (51%) · b1 | 1440 · 151.0 (82%) · b1 |
| 392k | — | — | — | — | 1184 · 181.4 (98%) · b1 |
| 400k | — | — | — | — | **OOM** |
| 480k | 975 · 116.9 (63%) · b1 | — | — | 990 · 141.1 (76%) · b1 | OOM |
| 600k | 655⚠ · 110.3 (60%) · b1 (anomaly; ~760 est) | — | — | 800 · 175.1 (95%) · b1 | OOM |
| 640k | **732** · 111.4 (60%) · b1 | — | — | 731 · 181.5 (98%) · b1 | OOM |
| 660k | — | — | — | **OOM** | OOM |
| 800k | **597** · 147.5 (80%) · b1 | — | — | OOM | OOM |
| 900k | **519** · 183.0 (99%) · b1 | — | — | OOM | OOM |
| 1.1M | (doesn't fit) | pred 436 · ~158 (85%) · b1 (run pending) | — | OOM | OOM |

### q3-32b (dense)

| ctx | asym T1-LATENCY | asym T3-MEMORY | so-unsloth | so-recomp |
|---|---|---|---|---|
| 49k | 3457 · 132.2 (71%) · b8 | 995 · 95.5 (52%) · b8 | 4344 · 177.3 (96%) · b8 | 2573 · 108.5 (59%) · b8 (so-off) |
| 80k | — | — | — | 1839 · 174.3 (94%) · b2 |
| 96k | — | — | 1674 · 145.2 (78%) · b3 | — |
| 128k | 1067 · 127.0 (68%) · b2 (post-fixes; was 957) | — | 1110 · ~123 (66%) · b2 | 1101 · 139 (75%) · b1 |

Reading down the columns: so-recomp dies first (392k fits → 400k OOM), so-unsloth
next (640k fits → 660k OOM), asym T1 runs to 900k at 99% and hands off to T2 at
1.1M; T3's 174k×b8 row is the deep big-batch capacity point. Q3-32b deep-end walls
were mapped on c12 (uns b1 edge 384k@98%; rc wall ∈(160k,192k]); c14 confirmed c12's
deepest capacity points to −0.4% with bit-identical reserved memory.

## Provenance
- Runs/tags: tput*-c14 in profiling_results/profiling_tp_s04-p1-dgx-02-c14/ (51+ dirs).
- Fix ladder (957→1067 dense, 948→975 MoE): fix_asym.md §5/§5a.
- Dial ladders and knob classes: scheduler_v2.md §3b/§7; unified formula §9.
- Scheduler code + selftest: scripts/lf/asym_scheduler.py (--selftest, --sweep).
- Interface/design history (β → budget → allocation): agent/handoffs/prompt.md.
- Remaining work: agent/impls/remaining_optimizations.md.
