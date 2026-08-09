# Placeholder-fill campaign — POINT OF RECORD (started 2026-07-28, node s04-p1-dgx-02-c18)

USER GOAL (2026-07-28): fill in every missing/placeholder (est-bordered) cell in the
combined throughput plots — q3.5-35b ranks 1+2 FIRST, then q3.5-122b ranks 1+2, then
all other models' est cells in the combined figures. Don't stop until done. User
waived same-node discipline ("very little deviation between machines — ok for this
machine to keep filling"); node is in fact c18 = the same node as the q3.5 campaign
tables, so all new cells are same-node with the existing q3.5 rows and cross-node
only vs the c12/c14-era 30b/32b/llama cells.

Tree: `main_kevin` @ d22440d (same tree as the closed q3.5 campaigns —
fix_qwen3.5_tp.md §7-§10). Protocol: w1+m2 (PROFILERS=source MAX_STEPS=2
WARMUP_STEPS=1), MAX_SAMPLES=1024 (n512 at ≥900k), one run at a time, GPU-empty +
MemAvailable≥1200GB(fabric) + stale-arena guards before every launch, kill -9 exact
PID only. Rank-2: GPUs 0+1, b1/rank ga1, GLOBAL TP = 2× per-invocation parse;
cell = lat s/it · TP tok/s · HBM GiB (%) · RSS GB (per-rank at rank 2).

Tooling (this campaign): `scripts/lf/tp_probe_fill.sh` (rank-aware tp_probe with
artifacts-FIRST verdicts + freshness-bounded evidence + one-shot DATASET_OVERWRITE
retry), `scripts/lf/parse_fill_cell.py` (cell extractor: lat.md e2e-measured-step;
summary.md whole-process peak reserved HBM / RSS bytes), chain scripts + logs +
results.tsv under `/scratch_local/user_data/shutian/kevin/cache/fillruns/`
(host+container shared). Launched from host via `enroot start` into asym_sft_46.

## §0 EST-CELL INVENTORY (from plot_tp_vs_seq.py + plot_tp_vs_seq_2r.py DATA, 2026-07-28)

RANK-1 (plot_tp_vs_seq.py): q3.5-35b ZERO est · q3.5-122b ZERO est (both fully
measured on c18 — nothing to do). q3-30b: recomp@80k 4400 · uns-OFF@128k 2650 /
@320k 1290 / @480k 905 / @640k 690 · asym T1@128k 2950. q3-32b: asym T1@320k 528 ·
uns-OFF@192k 577 / @320k 419 / @384k 350 / @448k 300 / @576k 234 + est host wall
~600k (640k OOM* never probed). llama3.3-70b: uns-OFF@96k 603 / @320k 306 /
@352k 288 / @384k 272 + est wall ~416k (416k/448k OOM* never probed) · asym
T2@352k 355 (host-OOM flake, never re-probed) · asym T2@416k 299 (run completed,
profiler artifacts lost to the 401 crash — steady from 3-step total).

RANK-2 (plot_tp_vs_seq_2r.py): q3.5-35b: uns@256k 1507 / @384k 1771 ·
uns-OFF@256k 1454 / @384k 1726 · asym sEP-T2@256k 1980 / @384k 2510 / @576k 2770
(interp) + uns-OFF@576k OOM* inside its (512k,640k] wall bracket (probe-able).
q3.5-122b: uns@128k 1374 (ratio-est) — plus two upgrade-grade gaps: asym@128k
measured only at b1 (950 @63% HBM — squeeze decides the 128k sign vs uns) and
uns-OFF row is OOM* from rank-1 inference (one measured rank-2 confirm upgrades it).
q3-30b: uns@384k 2375 · asym sEP-T2@384k 2450. q3-32b: rc@128k 2080 · uns@128k
2080 / @168k 1745 / @256k 1265. llama: uns@104k 1680 / @168k 1260.
NOT in scope as measurements: FSDP2/ZeRO3 series (derived-by-design from recomp ×
deterministic multipliers in both scripts — filling them means running real FSDP2/
ZeRO3 baselines; flagged to user as a scope decision, not silently attempted).

Weights on c18: 35b/122b/30b/32b cached ✓; **llama3.3-70b NOT cached — download
(~140 GB) queued before its phase.**

## §1 RUN QUEUE (serial; updated as verdicts land)

- **Phase A (35b R2)**: 256k uns/uns-OFF/sEP-T2 → 384k uns/uns-OFF/sEP-T2 →
  576k sEP-T2 → 576k uns-OFF wall probe (expect HOST-C-OOM). [chainA]
- **Phase B (122b R2)**: uns@128k (b2→b1 best-batch) · sEP-T1@128k b2 (squeeze) ·
  uns-OFF@128k confirm. Arena 400 for sEP.
- **Phase C (R1)**: 30b (rc@80k, uns-OFF@128k/320k/480k/640k, asym T1@128k) →
  32b (asym T1@320k, uns-OFF@192k/320k/384k/448k/576k, uns-OFF@640k wall probe) →
  llama (weights download; uns-OFF@96k/320k/352k/384k, uns-OFF@416k wall probe,
  asym T2@352k re-probe, asym T2@416k re-measure).
- **Phase D (R2)**: 30b (uns@384k, sEP-T2@384k) → 32b (rc@128k, uns@128k/168k/256k)
  → llama (uns@104k, uns@168k).

## §2 PROGRESS LOG (newest at bottom; every run and verdict)

- [2026-07-28] Campaign opened. GPUs empty, no live runs (both prior chats idle);
  /dev/shm had two stale asym_fabric_33247 remnants (64K done + 128K manifest, no
  arena payload) — cleared by chainA guard at launch. Est inventory (§0) built from
  the two DATA dicts. chainA (Phase A, 8 runs) LAUNCHED [10:53Z].
- [2026-07-28] ~~LLAMA BLOCKED ON HF TOKEN~~ **RESOLVED [00:1xZ 07-29]**: token
  was in env/bashrc.sh:97 all along (user pointer; the earlier sweep grep'd only
  export.sh/setup.sh — recorded as a miss). Token written to
  cache/huggingface/token (0600); **Llama-3.3-70B-Instruct snapshot download
  STARTED in background** (safetensors only, original/*.pth excluded, ~132 GB;
  parallel to chain C — network+disk only). chainF (9 llama cells: R1 uns-OFF
  96k b3-ff + 320k/352k/384k b1 + 416k wall probe + asym-T2 352k re-probe/416k
  re-measure; R2 uns 104k/168k b2-ff) staged — runs after chainE.
- [2026-07-28] Batch conventions for fill cells settled from the c12/c14 records
  (cells must match their row's convention): 30b 80k=b8-class first-fit (rc
  b8→6→5→4), 128k best-batch (uns was b4@81%, rc b2 → uns-OFF b8→6→4, asym T1
  b6→5→4), 320k+ b1; 32b 192k b2→b1 (160k measured b2), 320k+ b1, asym T1@320k
  b1 (parity-band cell); 32b R2 uns@128k/@168k b2→b1 (rank-1 uns@128k was b2),
  256k b1; 30b R2 384k b1 (row is b1-ladder); llama 96k b3→b2 when unblocked.
  chainB (122b R2) / chainC (30b+32b R1) / chainD (30b+32b R2) scripts staged in
  fillruns/, launch serially in that order after chainA.

## §3 MEASURED CELLS (cell = lat s/it · GLOBAL TP tok/s · HBM GiB (%) · RSS GB(/rank at r2) · spread%)

| model | rank | seq | system | was (est) | MEASURED | verdict |
|---|---|---|---|---|---|---|
| q3.5-35b | 2 | 256k | uns | 1507 | 323.3 · **1584** · 94.7 (51%) · 143 · 0.7% | FIT b1 (+5.1% vs est) |
| q3.5-35b | 2 | 256k | uns-OFF | 1454 | 341.9 · **1498** · 46.0 (25%) · 262 · 0.9% | FIT b1 (+3.0%) |
| q3.5-35b | 2 | 256k | asym sEP-T2 | 1980 | 255.3 · **2005** · 78.8 (43%) · 273 · 1.0% | FIT b1 (+1.3%) — **+27% h2h** |
| q3.5-35b | 2 | 384k | uns | 1771 | 427.1 · **1798** · 136.8 (74%) · 191 · 1.8% | FIT b1 (+1.5%) |
| q3.5-35b | 2 | 384k | uns-OFF | 1726 | 450.3 · **1706** · 73.4 (40%) · 392 · 1.5% | FIT b1 (−1.2%) |
| q3.5-35b | 2 | 384k | asym sEP-T2 | 2510 | 311.3 · **2467** · 103.4 (56%) · 328 · 0.5% | FIT b1 (−1.7%) — **+37% h2h** |
| q3.5-35b | 2 | 576k | asym sEP-T2 | 2770 interp | 426.1 · **2704** · 136.3 (74%) · 439 · 0.2% | FIT b1 (−2.4%) |
| q3.5-35b | 2 | 576k | uns-OFF | OOM* (wall (512k,640k]) | **HOST-C-OOM** | **wall TIGHTENED to (512k,576k] — asym sole coverage now starts AT 576k** |
| q3.5-122b | 2 | 128k | uns | 1374 | 354.3 · **1445** · 168.9 (91%) · 399 · 0.4% | FIT b2 (+5.2% vs est) |
| q3.5-122b | 2 | 128k | asym sEP-T1 | 950 (b1 only) | 307.5 · **1665** · 179.6 (97%) · 542 · 0.4% | FIT b2 squeeze — **column flips to asym +15%** (b1 would have read −34%) |
| q3.5-122b | 2 | 128k | uns-OFF | OOM* (rank-1 inference) | **HOST-C-OOM** b1 | measured at its shallowest point — row now wall-anchored |
| q3-30b | 1 | 80k | rc | 4400 | 61.5 · **5206** · 150.5 (81%) · 175 · 0.2% | FIT b4 (b8/6/5 OOM) — ⚠️ chain-E re-anchor pending |
| q3-30b | 1 | 128k | uns-OFF | 2650 | 503.9 · **2032** · 149.9 (81%) · 662 · 0.1% | FIT b8 (est was 23% optimistic) |
| q3-30b | 1 | 128k | asym T1 | 2950 | 257.6 · **2982** · 182.5 (99%) · 432 · 0.2% | FIT b6 (est +1.1% accurate; ≈parity with banked uns 3055) |
| q3-30b | 1 | 320k | uns-OFF | 1290 | 259.4 · **1234** · 49.8 (27%) · 403 · 0.0% | FIT b1 (est −4.3%) |
| q3-30b | 1 | 480k | uns-OFF | 905 | 540.1 · **889** · 73.4 (40%) · 420 · 0.1% | FIT b1 (est −1.8%) |
| q3-30b | 1 | 640k | uns-OFF | 690 | 954.9 · **670** · 98.8 (53%) · 629 · 0.1% | FIT b1 (est −2.9%) — 30b R1 fills COMPLETE |
| q3-32b | 1 | 320k | asym T1 | 528 | 609.7 · **525** · 145.8 (79%) · 424 · 0.1% | FIT b1 (est −0.6%; parity with banked uns 531 ✓) |
| q3-32b | 1 | 192k | uns-OFF | 577 | 682.4 · **563** · 117.0 (63%) · 691 · 0.1% | FIT b2 (est −2.4%) |
| q3-32b | 1 | 320k | uns-OFF | 419 | 779.1 · **411** · 98.2 (53%) · 605 · 0.1% | FIT b1 (est −1.9%) |
| q3-32b | 1 | 384k | uns-OFF | 350 | 1063.9 · **361** · 117.0 (63%) · 691 · 0.3% | FIT b1 (est +3.1%) |
| q3-32b | 1 | 448k | uns-OFF | 300 | **HOST-C-OOM** b1 | **WALL FOUND (384k,448k] — est wall ~600k was 150k+ too optimistic**; 576k/640k cells now beyond a MEASURED wall |

| q3-30b | 2 | 384k | uns | 2375 | 324.5 · **2367** · 114.7 (62%) · 184 · 0.2% | FIT b1 (est −0.3%) |
| q3-30b | 2 | 384k | asym sEP-T2 | 2450 | 331.8 · **2314** · 92.0 (50%) · 355 · 0.2% | FIT b1 (est −5.6%; measured column = rc 2370 / uns 2367 / asym 2314 @ −12pp HBM — convergence, asym edge starts 640k) |
| q3-32b | 2 | 128k | rc | 2080 | 120.1 · **2131** · 139.1 (75%) · 48 · 0.2% | FIT b1 (est +2.5%; DP ratio 1.94) |
| q3-32b | 2 | 128k | uns | 2080 | 238.3 · **2149** · 127.9 (69%) · 323 · 0.1% | FIT b2 (est +3.3%; rc≈uns ✓, asym 2256 leads +5%) |
| q3-32b | 2 | 168k | uns | 1745 | 383.6 · **1752** · 166.6 (90%) · 323 · 0.1% | FIT b2 (est +0.4%; asym 1797 leads +2.6%) |
| q3-32b | 2 | 256k | uns | 1265 | 407.3 · **1257** · 127.9 (69%) · 323 · 0.2% | FIT b1 (est −0.6%; asym 1305 leads +3.8%) — CHAIN-D COMPLETE 6/6 |
| llama | 1 | 96k | uns-OFF | 603 | 314.3 · **611** · 73.6 (40%) · 705 · 0.3% | FIT b2 (b3 OOM; est +1.3%) |
| llama | 1 | 320k | uns-OFF | 306 | **HOST-C-OOM** b1 | **WALL (192k,320k] — est wall ~416k was ~100k+ too deep**; 352k/384k est cells + 416k probe MOOT (cancelled; chain restarted trimmed) |
| llama | 1 | 352k | asym T2 | 355 (flake-est) | **HOST-C-OOM** b1 | the c12-era "host-OOM flake" REPRODUCES on c18 — not a flake. 416k re-measure now DIAGNOSTIC: fit ⇒ real mid-band host anomaly (122b c14 288-320k class); C-OOM ⇒ c18 host-tighter for llama T2 deep-end (c12 384k/448k cells node-inconsistent) |
| llama | 1 | 416k | asym T2 | 299 (artifacts-lost est) | 1385.4 · **300** · 182.2 (98%) · 989 · 1.4% | FIT b1 (est +0.3% exact) — **ANOMALY CONFIRMED REAL: 352k C-OOM with 384k/416k/448k fits = mid-band host-DNF band, 122b-c14 class.** 352k third-strike retry appended to E2; cell goes red-OOM if it C-OOMs again |
| llama | 2 | 104k | uns | 1680 | 244.5 · **1701** · 120.5 (65%) · 435 · 0.4% | FIT b2 (est +1.3%; asym 1789 leads +5.2%) |
| llama | 2 | 168k | uns | 1260 | 267.8 · **1255** · 97.0 (52%) · 435 · 0.5% | FIT b1 (b2 OOM; est −0.4%; asym 1317 leads +4.9%) — **CHAIN-F COMPLETE** |
| q3-30b | 1 | 80k | uns (E2 b6) | c14 3424 b8-edge | 90.8 · **5288** · 140.9 (76%) · 278 · 0.1% | b6 beats b8-edge 4335 by +22% — **rc≈uns RESTORED (5206/5288, −1.6%)**; c14 80k cells were edge-taxed |
| q3-30b | 1 | 80k | uns (E2 b4) | — | 62.0 · 5164 · 94.8 (51%) · 278 · 0.2% | curve flat-tops at b6 (5164→5288→4335-edge) |
| q3-30b | 1 | 80k | asym T1 (E2 b8) | c14 3642 | 128.4 · **4984** · 153.1 (83%) · 432 · 0.2% | b8≈b6 (4984/4983) — asym max found. **c18 column: asym −5.7% vs uns 5288; c14's "P1 asym +6%" was uns-edge-tax artifact** (complete-figure only; 80k LEAN_DROPped) |

- [03:4xZ] chain-E e1: **c18 uns @80k = 147.6 · 4335 · 180.1 (97%) · 382 · 0.5%
  FIT b8** — +27% over the c14-banked 3424 (186.9 s/it, 96%): the c14 cell was
  EDGE-TAXED (fragile-edge class), not a node constant. rc 5206 (b4, 81%) is
  still +20% over c18-uns-b8 → rc≈uns says uns's best batch on c18 may not be
  b8. E2 QUEUED (after chain F): uns @80k b6 → b4 (max-batch probes) +
  uns-OFF @80k b8-ff → completes a same-node best-batch-honest 80k column.
  NOTE: 80k is LEAN_DROPped from the 30b paper figure — this column is
  tp_complete-only; asym re-anchor (e2) in flight.
- [08:35Z 07-29] **CHAIN-E2 DONE — CAMPAIGN CLOSED.** E2 verdicts: uns @80k b6
  = 5288 (76%) / b4 = 5164 (51%) → uns max = b6 5288 (its b8 4335 was the edge
  cell); asym T1 b8 = 4984 (83%) ≈ b6 4983 → asym max found; uns-OFF b8 = 2693
  (53%; c14 2806 −4%, cross-node consistent); llama T2@352k RETRY = **FIT 349
  b1 (89%, 0.1% spread; RSS 960 ≈ the ~957 GB budget)** — the cell is a HOST
  KNIFE-EDGE (2 C-OOMs + 1 fit across attempts), recorded as measured 349 with
  this fragility note; the "mid-band anomaly" hypothesis is retired (416k's
  RSS 989 fitting twice shows host cost is pool-shape-, not strictly seq-,
  monotone). 30b 80k column re-anchored wholesale to c18 (rc 5206 · uns 5288 ·
  uns-OFF 2693 · asym 4984): c14's "P1 asym +6%" verdict was an uns-edge-tax
  artifact; honest short-end reads asym −5.7% (complete-figure only, 80k is
  LEAN_DROPped). FINAL AUDIT: zero "est" data cells remain in either DATA dict
  (FSDP2/ZeRO3 stay derived-by-design per the plot's own convention); all 28
  figures (16 R1 + 12 R2) regenerated from fully-measured rows.

## §5 SATURATION + CONTENTION REDO (2026-07-29, user directives)

USER (07-29): (1) 35b R1 128k/384k look undersaturated — probe better asym
batches; 122b R1 asym 32k/128k too. (2) 122b 32k/128k OOMs (esp. uns-OFF) are
suspicious — "weird contentions"; retest one-at-a-time on clean GPUs 0/3 with
proper NUMA pinning. (3) 122b R2 is weak — configure so an asym tier clears
320k and beyond. Don't stop until met.

DIAGNOSIS (before reruns): GPUs showed 0.7/11.4/12.0/8.3 GB "used" with ZERO
processes — stray driver-held regions on the GPU HBM NUMA nodes (nodes
2/10/18/26; CPU nodes are 0,1 with ~957GB). The TRAINER runs were membind
0,1 (train.log: NUMACTL_ENABLE=1 MEMBIND=0,1 — driver default, verified),
but auxiliary steps (dataset builders etc.) run unpinned, and the residues
accumulated during the fill campaign. drop_caches: no privilege; node-pressure
touch: ineffective (not page cache); **CUDA allocation pressure WORKED**:
claim-all-free + empty_cache per GPU released most of it — floors now
0.7/3.7/3.8/3.6 GB, every GPU verified able to hand out 178-182 GiB. Note the
banked R2 cells hit 179.6 GiB reserved on GPU1 (impossible with 11.5 stuck),
so the residue postdates those runs — contamination plausible for LATER runs
and for edge cells; hence the retests. Singles → GPU0 (cleanest; user's 0/3
choice honored for singles); pairs → (0,1) because the driver hard-requires
same-superchip pairs (0,3 rejected: "must use --gpus 0,1 or 2,3").

Chain G (35b R1): asym T2 128k b9→b8 (banked 2484@b8; up-probe) · asym T2
384k b3→b2 (b3 NEVER probed; banked 1801@b2) · uns-OFF 128k b9 (b9 pred 98%,
skipped by the old ≤92% rule). Chain H (122b R1): rc 32k b6→b5 + 128k b2
(rc rows sit at 63% HBM — undersaturated) · asym T1 32k b14→b13 + 128k b4
(97% rows — honest up-probes) · uns 128k b3 · uns-OFF 32k b8 RETEST (was
HOST-C-OOM ×2) + 128k b1 (never probed at rank 1 — row shows OOM*). Chain I
(122b R2, arena 400→340): decision tree sEP-T2@320k retest → T1+ohbm8 →
T1+ohbm16; winner walks 352k→384k. All runs: NUMACTL asserted + PIN-VERIFIED
grep per run + GPU-residual line in the log. results_redo.tsv.

### §5 CONCLUSIONS (2026-07-30, 17 redo runs, campaign closed)

1. **Saturation (goal 1): the flagged cells were already at their peaks** —
   35b 128k b8 2484 (b9 fits-but-slower 2229), 384k plateau b3 1810 ≈ b2 1801
   (cell → 1810, the only value change); 122b asym 32k b12 1155 / 128k b3
   1042 confirmed by G-OOM up-probes.
2. **Weird OOMs (goal 2): two real defects found and fixed** — rc @32k was
   31% undersold (475 b4@63% → **682 b6@91%**, b7 G-OOM); uns-OFF's all-dead
   row was a ×b8-cluster artifact: **403 @32k (b4) and 377 @128k (b1) FIT**
   (RSS ~705 GB); its b8-cluster + ≥288k host-deaths reconfirmed physical
   (3rd strike, clean env). 122b 32k column final: rc 682 · uns 961 ·
   uns-OFF 403 · asym 1155.
3. **Contention/NUMA: real but bounded** — 8-12 GB stray driver-held HBM on
   GPUs 1-3 (zero processes) released via CUDA allocation pressure (floors
   ~3.7 GB); trainers were always membind 0,1 (now PIN-VERIFIED per run);
   redo ran one-at-a-time on GPU0 / pair (0,1). No banked verdict flipped
   due to contention alone — every OOM retested reproduced except the
   uns-OFF small-batch cells, which were never-probed rather than corrupted.
4. **122b R2 past 320k (goal 3): NOT achievable on this node class — a
   characterized limit, all sanctioned levers exhausted**: T2/T2B/T3 +
   off-recompute HOST-C-OOM (arena ~341 GiB + duplicated per-rank pools ≫
   957 GB); T1 G-OOM at 320k (recompute working set + DP overhead over a
   rank-1 cell that already sat at 88-89%); delta-chunk 8000 insufficient;
   ohbm dials are keep-dials (ohbm0 = already max-offloaded). The rank-2
   corridor is closed from both ends — llama-R2A class. The POSITIVE R2
   verdicts stand: +15% @128k (best-batch), +31% @288k, capacity tie at
   (288k,320k]. Lever for bigger-host nodes remains #7b (linear-attn
   keep-acts port).

## §6 122b R2 CAPACITY PUSH v2 — **GOAL ACHIEVED 2026-07-30 15:5xZ** (user:
## "more HBM, less host — make it past the bar")

**fixr1: asym sEP-T1 + GRAD-ONLY offload @320k RANK-2 = FIT — 408.9 s/it ·
1565 GLOBAL tok/s · 159.9 GiB (86%) · RSS 562/rank · 0.6% spread.** The
(288k,320k] capacity tie is BROKEN: uns G-OOMs at 320k, asym trains it —
sole coverage begins at 320k, at HIGHER TP than its own 288k cell (1565 vs
1542: seq amortization absorbs the sync-D2H cost). Winning stack:
asym_sepplan2 T1 (unsloth-ohbm16) + ASYM_EP2_GRAD_OFFLOAD=true (the new
CPU-flat allreduce port) + ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0 (sync D2H —
the async path's keepalives defeat the shed) + WEIGHT offload OFF (its pinned
copies were self-defeating on this host-bound wall) + arena 345 + floor 40
(within c14's 35 precedent). **NEW WALL = (320k, 352k] HOST-C-OOM, measured
×2 (ohbm16 AND ohbm8 both die at 352k) — one full rung past the old tie.**
FINAL 122b R2 ROW: 1665@128k (+15%) · 1542@288k (+31%) · **1565@320k SOLE** ·
352k wall. DATA updated, 12 tp2r figures regenerated. The §4/§8 "capacity
tie" verdicts are SUPERSEDED (addendum in fix_qwen3.5_tp.md).

### §6b CEILING SQUEEZE (2026-07-30 20:4xZ+, user: "86% HBM — push higher")

352k attack map (grad-only stack throughout): floor-40 dips 38/38 (r2 ohbm16,
r3 ohbm8) → **s1 ohbm16@floor35 HOST-C-OOM fired 34** (dip deepens as floors
drop — true bottom ≤34) · **s3 ohbm4@floor35 = G-OOM** (the 1/4 root-keep
overshoots HBM at 352k). 352k is PINCERED: too little keep → host short ~1-2
GiB; too much keep → GPU over. CHAIN-T: the untried middle — **ohbm8@floor35**
(t1; host relief ~+2 over ohbm16, GPU between the pincers ~94-96%) → fit walks
384k (t2) · fail → t3 = ohbm8@floor30, FLAGGED: below the c14-35 precedent,
30 GiB kernel cushion, one-run capacity probe only.

**352k VERDICT: OVER-BUDGET, CONCLUSIVE** — t1 (ohbm8@35) fired 34; t3
(ohbm8@30, flagged) fired **29 < 30**: the dip bottoms ~1 GiB under EVERY
floor (38→34→29 across 40/35/30) = demand exceeds the node outright, no floor
saves it; ohbm4 G-OOMs the GPU side. 5 attempts, both walls bracketed.
**CHAIN-U FINE-RUNG: 336k ohbm16@floor35 = FIT — 448.7 · 1498 GLOBAL ·
166.8 (90%) · RSS 558/rank · 0.6%.** (Ops: chain-U first launch re-ran t1 via
a sed-of-sed ladder bug — killed by exact PID; a dead PID's un-reaped 1.7GB
driver context then tripped the guard → guard patched with a /proc-existence
stale-context filter; corrected probe ran clean.)

**§6 FINAL CEILING: asym 122b R2 = 336k tokens (SOLE at 320k AND 336k), wall
(336k, 352k] host-bound.** Row: 1665@128k (+15%) · 1542@288k (+31%) ·
1565@320k SOLE · **1498@336k SOLE crown (1.17× past uns's 288k last fit)** ·
352k wall ×5-measured.

MEMORY DIFF (per-component peak breakdown, T1@288k, r1 q122d1t1288 vs r2
q2t288t1): rank-2 allocated peak 170.9 vs rank-1 134.6 GiB; the delta is
**routed_experts trainable weights +18.0 GiB and grads +18.0 GiB resident in
GPU HBM at rank 2** (0-byte GPU placeholders at rank 1; activations identical
73 GiB). Root cause: run_lf_lora_sft.sh's EP2-family branch HARD-FORCED
ASYM_CPU_ADAMW_GRAD_OFFLOAD=false + WEIGHT_OFFLOAD=false (hook-offload vs
manual-allreduce ordering hazard), while rank-1 122b T1 ran gradofftrue —
the rank-2 HBM deficit at 320k (~10 GiB) is entirely this forced residency.
FIX: driver patched — ASYM_EP2_WEIGHT_OFFLOAD / ASYM_EP2_GRAD_OFFLOAD env
overrides (defaults unchanged false). WEIGHT side is hazard-free (weights are
never allreduced; bf16 copies live pinned host-side exactly as rank-1); GRAD
side stays off (hook D2H would ship pre-reduction grads — needs a
post-allreduce port if ever wanted). Host cost of weight-offload ≈ +18 GB/rank
— trivially inside budget for host-lean T1 (RSS 543/rank at 288k).
CHAIN-L: l0 64k smoke (WOFF-VERIFIED + loss-parity vs banked q2r1sep64 +
HBM-drop check) → l1 320k tie-breaker → l2 352k → l3 384k.

### §6 verdicts (as they land)

- fixl0 (weight-offload ONLY) HARDFAIL by design: LF parser couples the flags
  (`weight_offload=true requires grad_offload=true` — they share the
  post-accumulate hook). And grad-offload under EP2 has a REAL hazard, read
  from the code: hooks D2H each grad into the optimizer's pinned fp32 flat
  buffer and NULL param.grad during backward → the GPU flat reduce would
  average ZEROS while the CPU optimizer steps unreduced LOCAL grads — silent
  rank divergence (this is exactly what the forced-false guarded).
- **PORT SHIPPED (3 files)**: (1) run_lf_profiled_train.py `_ep2_post_backward`
  now detects an offloaded AsymCPUAdamW and reduces the CPU FLAT BUFFER
  instead: drain → zero stale per-mapping regions (zeros-materialization
  semantics preserved) → chunked H2D→all_reduce→mean→D2H via one persistent
  1-GiB GPU stage (env ASYM_EP2_CPU_REDUCE_BUCKET_MB) → mark all mappings
  reduced (both ranks step identical averaged grads; masters lockstep; also
  skips building the GPU flat bucket entirely). Vanilla-EP owner-scaled params
  rejected loudly. (2) LF parser: the two distributed guards accept the
  explicit ASYM_DP=1+ASYM_EP2=1+ASYM_EP2_GRAD_OFFLOAD opt-in (everything else
  keeps the hard error). (3) run_lf_lora_sft.sh EP2 branch maps
  ASYM_EP2_GRAD/WEIGHT_OFFLOAD → the adamw flags (defaults false, unchanged).
- CHAIN-M LAUNCHED: m0 64k smoke (GOFF-VERIFIED + losses vs banked
  0.9787/1.0992/0.9819 + HBM/RSS delta) → m1 320k → m2 352k → m3 384k →
  m4 416k. Expected relief ≈ 36 GiB HBM/rank (18 weights via coordinator
  release + 18 grads via during-backward hook streaming).
- [13:5xZ] **m0 SMOKE PASS**: fixm0 @64k R2 = 258.6 · 495 global · **50.8 GiB
  (27%) — banked smoke was 81.6 GiB: −29.6 GiB confirms grad+weight offload +
  CPU-flat reduce ACTIVE**; losses 0.9783/1.1017/0.9881 vs banked
  0.9787/1.0992/0.9819 (max Δ 0.006, bf16-noise class, within the 0.02 gate);
  RSS 547/rank (grad flat + staging absorbed fine). Verification notes: the
  run-dir 'gradofffalse' token is the PROFILE driver's label composed before
  run_lf's EP2 branch override — cosmetic mislabel only (run_lf log_kv shows
  =true); heartbeats don't land in train.log (separate stream) — HBM delta is
  the authoritative proof. m1 @320k in flight.
- [14:1xZ] m1 @320k = **HOST-C-OOM by ~1 GiB** (watchdog: avail 49 < floor 50)
  — the offload's HOST cost (fp32 grad flat ~36 + bf16 async staging ~18 +
  pinned bf16 weights ~18 GB/rank) ate exactly the margin the HBM side gained.
  RECOVERABLE SLACK identified: arena cap 400 vs measured need ~341 (the
  /dev/shm file is committed at CAP size → trim to 345 reclaims ~55 GB) and
  the bf16 staging is droppable (ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0 → sync
  per-param D2H, slower step, −18 GB/rank). CHAIN-N: same ladder with arena
  345 + sync D2H (net ≈ +91 GB host vs m1's attempt, ~90× the observed
  shortfall). GPU-side these levers change nothing (−36 GiB HBM stands).
- [14:2xZ] n0 smoke: **sync-D2H is strictly better** — 36.7 GiB HBM (vs m0's
  50.8: the ASYNC path keeps every CUDA grad source alive until the per-step
  drain, defeating the shed; sync frees at copy time), TP 494≈495 (cost
  invisible at 64k), RSS 515 (staging gone), losses in band. n1 @320k =
  HOST-C-OOM at 46 GiB avail (floor 50) with arena-345 VERIFIED on disk
  (file exactly 345 GiB) — remaining seq-scaling host block = T1's checkpoint
  ROOTS (ohbm0 sends 100% to CPU per rank). CHAIN-O: the user's exact "more
  HBM less host" dial now that HBM has −36 GiB slack — unsloth-ohbm16 keeps
  every 16th root in HBM (~8-12 GiB HBM/rank ⇄ ~16-25 GB node host relief vs
  a ~4 GiB shortfall). Ladder: ohbm16@320k → 352k (ohbm8 host-fallback) →
  384k. (fixi2/i3's ohbm failure is explained: the keep-dial had no HBM room
  before the offload port existed.)
- [15:1xZ] q1 @320k (floor 35 = c14 protocol) = HOST-C-OOM fired at **33 <
  35** — dip series across floors 50/40/35: 48 → 37 → 33: each floor cut just
  exposes more of a converging mid-step demand curve; with BOTH offloads the
  host is structurally over (weight-offload's +18 GB/rank pinned copies are
  self-defeating on this host-bound wall). FINAL COMBINATION (never tried):
  **GRAD-ONLY offload** — weights return to HBM (GPU → ~96% ceiling-hug, the
  regime this model's T2 cells repeatedly survived at 98%), host gets back
  36 GB node vs a dip-shortfall of ~2-5. CHAIN-R launched (grad-only + sync
  D2H + arena 345 + floor 40 + ohbm16).
- [14:5xZ] p1 @320k (floor 40) = HOST-C-OOM, fired at **37 < 40** — the dip is
  deeper than o1's sampled 48 (the watchdog samples; with a lower floor the
  run survives longer and the dip's true depth shows). 37 clears floor 35 by
  2 GiB → CHAIN-Q (floor 35 = the EXACT c14 122b protocol: its banked uns-OFF
  verdicts read "watchdog 34 < 35") launched — the last environmental attempt
  before characterization.
- [14:36Z] o1 @320k ohbm16 = HOST-C-OOM at **48** GiB (n1 was 46: ohbm16
  recovered ~2 GB — roots are NOT the dominant block). DECISIVE ARTIFACT:
  process_memory.csv shows step-boundary avail **79-96 GB** — the run is
  ~50 GB above water at equilibrium; only a mid-step transient (backward-peak
  host churn) dips under the 50-GiB floor, by 2 GiB. The floor is per-model
  config (WATCHDOG_FLOOR_GB_BY_MODEL / HOST_MEM_WATCHDOG_FLOOR_GB) and the
  c14 122b campaign ran floor=35 (banked "watchdog 34 < 35" verdicts) — 50 is
  a conservative setting, not physics. CHAIN-P: identical full stack + floor
  40 (still +5 over the c14 precedent; o1's dip 48 clears by 8). Ladder
  320k → 352k (ohbm8 fallback) → 384k.

### §5 verdicts (as they land)

- fixg1 **35b R1 asym T2 @128k b9 = 516.9 · 2229 · 181.0 (98%) · 724 · 3.7% FIT**
  (PIN-VERIFIED; dataset self-heal fired once). b9 < b8: batch curve = 2447(b7)
  → **2484(b8 peak)** → 2229(b9 edge-taxed). 128k cell CONFIRMED saturated at
  b8 2484 — not undersaturated.
- fixg2 **35b R1 asym T2 @384k b3 = 636.4 · 1810 · 182.9 (99%) · 788 · 4.5% FIT**
  (PIN-VERIFIED). b3 ≈ b2 (1810 vs 1801, +0.5% = noise): 384k plateau found,
  b2 was ~saturated. Cell → 1810 per max-measured rule (b2 1801@92% = the
  healthy-batch alternative, kept in comment).
- fixg3 35b R1 uns-OFF @128k b9 = **HOST-C-OOM** → b8 2020 (88%) confirmed its
  max. **CHAIN-G DONE: the user-flagged 128k/384k columns were NOT
  undersaturated — peaks verified by up-probes on the clean pinned env.**
  DATA 384k cell 1801→1810.
- fixh1 **122b R1 rc @32k b6 = 281.5 · 682 · 168.3 (91%) · 553 · 0.7% FIT**
  (PIN-VERIFIED) — **+44% over the banked b4 475 (63%): rc WAS undersaturated
  exactly as the user suspected.** Cell → 682. (b7 top-off probe → loose-ends
  queue; b6 at 91% leaves modest headroom.)
- fixh2 122b R1 rc @128k b2 = **G-OOM** → b1 412 CONFIRMED max (the 63% b1
  cell is real — rc's per-sample memory step doesn't fit twice at 128k).
- fixh3 122b R1 asym T1 @32k b14+b13 = **G-OOM both** → b12 1155 (97%)
  CONFIRMED max — asym 32k was already saturated.
- fixh4 122b R1 asym T1 @128k b4 = **G-OOM** → b3 1042 (97%) CONFIRMED max.
  Both user-flagged asym 122b cells were already at their peaks; the real
  undersaturation was rc@32k (fixh1, +44%).
- fixh5 122b R1 uns @128k b3 = **G-OOM** → b2 755 (92%) CONFIRMED max.
- fixh6 122b R1 uns-OFF @32k b8 RETEST = **HOST-C-OOM — 3rd strike (c14,
  c18-fill, c18-clean-redo)**: the verdict is PHYSICAL (234-GB weights
  streamed + act-offload pools ≫ 957GB host), NOT contention. The clean env
  did not change it. BUT →
- fixh7 **122b R1 uns-OFF @128k b1 = 339.3 · 377 · 53.9 (29%) · RSS 710 ·
  0.3% FIT** (PIN-VERIFIED) — **uns-OFF LIVES on 122b**: the "host-dead at
  every probed point" verdict was a b8-cluster artifact; b1's pools fit
  (710 < 957). Row updates: 128k OOM*→377 measured. Consequence: the 32k cell
  (only ever probed at ×b8) needs its own b-ladder (b6→b4→b2) — queued as
  chain J with the rc@32k b7 top-off. 288k b1 HOST-C-OOM (chain-16) stands as
  the wall. **CHAIN-H DONE (7/7).**
- chain-I first launch HARDFAILED at fixi1: my arena trim 400→340 was wrong —
  `shared fabric cap exceeded at 00640:bank (used=338.5 GiB, need +1.5)`;
  the "~310 GB bank" note was loose units, real need ≈ **341 GiB** → cap
  restored to the known-good 400 (leaked asym_fabric_41181 cleaned; GPU
  verified empty). Chain I+J RELAUNCHED [01:46Z] — fixi1 T2@320k retest in
  flight on pool 0,1, arena 400.
- fixi1 sEP-T2 @320k R2 RETEST (clean+pinned, arena 400) = **HOST-C-OOM —
  confirms the original verdict as PHYSICAL** (arena ~341 GiB + duplicated
  per-rank T2 pools ≫ 957GB), not contention. → fixi2 T1+ohbm8 dial next.
- fixi3 T1+ohbm16 @320k R2 = G-OOM (as predicted — keep-dial). CHAIN-I DONE:
  no dial in the wrong direction fits; chain K carries the corrected levers.
- fixj1 **122b R1 uns-OFF @32k b4 = 317.9 · 403 · 53.9 (29%) · 703 · 0.2%
  FIT** (b6 OOM) — the 32k cell flips measured-OOM(b8-only) → **403**. uns-OFF
  row final shape: 403@32k(b4) · 377@128k(b1) · C-OOM@288k — "host-dead
  everywhere" fully retired; it lives at small-batch short-seq and walls by
  288k (act pools scale it to death).
- fixj2 122b R1 rc @32k b7 = **G-OOM** → b6 682 (91%) CONFIRMED max.
  **CHAIN-J DONE.** 122b R1 32k column final: rc 682 · uns 961 · uns-OFF 403 ·
  asym 1155 — fully saturated + all four systems alive (was: rc undersold 31%,
  uns-OFF falsely all-dead). CHAIN-K LAUNCHED (waits for J teardown).
- [11:35Z 07-30] chain-K launch was DELAYED 8h by a self-matching pgrep in its
  wait-loop (launcher's own cmdline contained 'chainJ.sh') — killed, relaunched
  direct. fixk1 (T1 + QWEN35_DELTA_CHUNK_SIZE=8000 @320k R2) = **G-OOM** —
  chunk shed insufficient. fixk2 (sEP + unsloth-off-ohbm0 inner-act-offload
  recompute) in flight — the structural HBM shed; host bounded by shared arena.
- GRAMMAR LESSON (fixi2 G-OOM explained): **ohbm0 = ALL roots ALREADY
  offloaded to CPU; ohbmN KEEPS every Nth root in HBM** (driver docstring) —
  the llama "+ohbm12" precedent was a keep-dial on an -off- base, not a shed.
  fixi2/fixi3 therefore ADDED HBM vs plain T1 — wrong direction, both G-OOM
  by construction. chain K staged with correct levers: k1 = T1 +
  QWEN35_DELTA_CHUNK_SIZE=8000 (halve delta-net recompute working set — the
  rank-2 deficit over the 88-89% rank-1 cell is small); k2 = sEP +
  unsloth-OFF-ohbm0 (inner-act offload; host risk bounded since the arena is
  shared, unlike the dead uns-OFF baseline); winner walks 352k/384k.

## §7 R2 SIX-COLUMN COMPLETION (2026-07-31, user: 6 seqs per subplot, largest
## must hold asym)

After the record-vs-render rule (asym walls in DATA only, never plotted), four
lean panels fell under 6 columns: mixtral R1 (5), 32b R2 (5), llama R2 (5),
122b R2 (4). **Mixtral is NOT fillable from c18** — no model shorthand in this
tree's profile driver, no weights in the cache, campaign records (anchors_tmp)
live on c14 — flagged as the one 5-column exception (needs the c14 crew or a
model-integration port). CHAIN-V fills the other three with NEW MEASURED
columns at seqs asym holds (neighbor-consistent configs, no grad-offload —
all at/below the banked brackets): llama R2 +192k (uns, dp2-T1) · 32b R2
+320k (uns, dp2-T1→T2 fallback) · 122b R2 +192k,+256k (rc bracket-probe @192k
(wall was coarse (128k,288k]), uns, sEP-T1; rc@256k conditional on 192k fit).
Target lean rows: llama 104k/128k/168k/192k/224k/256k · 32b 128k/168k/256k/
320k/384k/416k · 122b 128k/192k/256k/288k/320k/336k.

### §7 verdicts (as they land)

- fv1 **llama R2 uns @192k = 330.9 · 1160 · 110.2 (60%) · 435 · 0.4% FIT b1**
  (monotone between 168k 1255 and 224k 1035 ✓).
- fv2 llama R2 asym dp2-T1 (plain ohbm0) @192k = **HOST-C-OOM** — instructive,
  not contradictory: the banked deeper cells (224k/256k) carry the **ohbm12
  root-keep dial** (llama's 80-layer roots ≈ the host hog; DATA comment
  documents "+ohbm12 dial" at 256k). Plain ohbm0 sends all roots host → dead
  at 192k. chain-W queued: same cell with unsloth-ohbm12 (row-consistent).
- fv3 **32b R2 uns @320k = 617.5 · 1036 · 158.8 (86%) · 323 · 0.2% FIT b1**
  (monotone 1257@256k → 1036 → 795@384k ✓).
- fv4 **32b R2 asym T1 @320k = 612.5 · 1045 · 138.7 (75%) · 426 · 0.2% FIT b1**
  — +0.9% over uns at −11pp HBM; T1 carries to 320k (shed point refines to
  (320k,384k] where the banked 384k is T2). No T2 fallback needed.
- fv5 **122b R2 rc @192k = 370.8 · 1035 · 168.3 (91%) · 296 · 1.1% FIT b1** —
  rc LIVES at 192k rank-2: wall bracket TIGHTENS (128k,288k] → (192k,288k];
  the conditional rc@256k probe (fv10) is now armed.
- fv6 **122b R2 uns @192k = 374.1 · 1026 · 135.3 (73%) · 399 · 0.6% FIT b1**
  (rc 1035 ≈ uns 1026: convergence law ✓).
- fv7 **122b R2 asym sEP-T1 @192k = 299.5 · 1282 · 145.7 (79%) · 544 · 0.5%
  FIT b1** — **+24% over the best baseline** in the new all-alive column
  (rc 1035 / uns 1026 / asym 1282).
- fv8 **122b R2 uns @256k = 446.7 · 1146 · 173.6 (94%) · 399 · 0.4% FIT b1**
  (uns rises 1026→1146→1178 toward its 288k edge — per-token amortization).
- fv9 **122b R2 asym sEP-T1 @256k = 339.7 · 1507 · 174.9 (95%) · 550 · 0.8%
  FIT b1** — **+32% over uns 1146**; asym R2 curve 1282→1507→1542→1565→1498
  rises into the 336k crown.
- fv10 122b R2 rc @256k = **G-OOM** ⇒ **rc R2 wall = TRUE bracket (192k,256k]**
  (fit 192k + OOM 256k both measured). **CHAIN-V DONE (10/10).** 32b + 122b
  R2 DATA rows updated (+320k · +192k/+256k) — both render exactly 6 lean
  columns ending at asym crowns.
- fw1 llama R2 dp2-T1+**ohbm12** @192k = **HOST-C-OOM** — the c14-banked row's
  cells (which fit 224k/256k with this dial) DO NOT reproduce on c18's host:
  another c14-vs-c18 divergence (fragile-cell class). CHAIN-X ladder: deeper
  root-keeps ohbm4 → (host-fail: ohbm2 | GPU-fail: ohbm6), cap 3 attempts,
  else llama stays 5 columns (flagged; row is c14-native).
- fx1 **llama R2 asym dp2-T1+ohbm4 @192k = 324.7 · 1183 · 151.9 (82%) · 526 ·
  0.1% FIT b1 — +2.0% over uns 1160**; ohbm4 trades T1's spare HBM for the
  80-layer root bytes (same still-T1 dial class as the banked 256k +ohbm12).

**§7 CLOSED (2026-07-31): every renderable panel = 6 measured columns ending
at asym's crown** — R2 rows finalized: 32b +320k · llama +192k · 122b
+192k/+256k (rc R2 wall = true bracket (192k,256k]); DATA + figures + Overleaf
ec7da49. Sole exception: mixtral R1 at 5 columns (not fillable from c18 — no
driver shorthand/weights/records here; c14-native row). A glm4.7-flash R2 row
appeared from the parallel campaign, already 6-column-conformant (the
structural filter applies to it automatically).

## §8 FULL DE-PLACEHOLDERING: OOM* SWEEP + ZERO3 + FSDP2 (2026-07-31, user)

USER: (a) asym's short-seq TP lead questioned → answered, no bug: max-TP-over-
batch convention + freed-HBM batch capacity; matched-batch parity measured
(122b 32k×b8 +0.9%; 35b 128k b4 1928-vs-1831). (b) ALL remaining black items
must become real: black OOMs (OOM*) AND the derived FSDP2/ZeRO3 bars. Order:
OOM sweep → ZeRO3-offload cells (driver already has zero3_offload_mem; wire
like superoffload if gaps) → FSDP2-offload (wire; ~equivalent offload set to
zero3-offload). Don't stop until no placeholders remain.

PLAN: chain Y = 36 OOM*-confirm runs (R1: 32b×2, llama×3, 122b×8, 35b×8;
R2: 35b×7, 122b×8; expected OOM, surprise fits banked). chain Z = ZeRO3 real
cells: zero3_offload_mem + recomp token at every rendered lean column where
recompute-class lives, batches seeded from rc's measured max (first-fit
descending), + first-OOM wall confirm per model/rank (~25 runs). Then FSDP2:
recon LF/HF fsdp plumbing, wire a fsdp2_offload backend (full-shard +
cpu-offload = zero3-offload-equivalent placement), loss-smoke, then the same
cell set as ZeRO3. Mixtral panel: still c18-unfillable (flag stands).

### §8 SWEEP COMPLETE (08-03 14:2xZ) — 36/36, zero surprise fits

Every rendered beyond-wall baseline cell in BOTH rank figures is now an
individually-probed, measured OOM: 20× G-OOM (incl. 2 NCCL@edge-class rank-2
deaths) + 16× HOST-C-OOM. The two flagged SIGKILL anomalies (fy2b640uns
attempt-1, fy1rc672 attempt-1) both re-ran to genuine CUDA-OOM verdicts —
transient external kills, not systemic. DATA dicts: 36 cells flipped
OOM*→OOM (residual OOM* markers exist ONLY at non-rendered record positions
+ the tp_complete_* mid-columns of 122b/35b — repo-side full-record variants,
NOT the paper figures; flagged as optional future probes). 34 figures
regenerated; Overleaf push e176565. Ops fixes that got the sweep through:
file-capture (orphan-pipe wedge), 3h timeout + orphan cleanup, reaper-guard
(teardown stragglers), log-and-continue on hardfail, train.log verdict
rescue, R2 early-terminate watchdog (saves ~2h per NCCL wedge; NOTE its
verdict-grep is fooled by stale verdicts in APPENDED logs — use fresh logs).
**CHAIN-Z (ZeRO3, 25 cells, Y5-hardened machinery) LAUNCHED 14:35Z**; GLM
weights downloading in parallel (GLM zero3 cells follow chainZ). FSDP2 =
other agent (run_baselines.md). (chainZ labels fits "SURPRISE-FIT" — Y-sweep
label semantics, cosmetic only; cell values parse normally.)

### §8 ZERO3 SERIES CLOSED (08-04 23:xx, Overleaf 5d8935f)

**ZeRO3-Offload is now a fully MEASURED series for all 7 runnable models,
both ranks** (mixtral = the standing derived exception): 52 fit cells + the
complete probe-everything OOM coverage (Z2: 37 confirms, all died as banked;
the 35b-256k est resolved to 844 measured; llama-128k OOM* resolved to
measured OOM → zero3 wall (96k,128k]). Verdict quality: qwen/llama fits
within ~±5% of rc (two cells step-identical — SuperOffload's extras are
noise at R2 b1); GLM fits +2-8% vs their c14-native rc rows (cross-node
band, user waiver). GLM enablement: liger glm4_moe port from the SFT-39
checkout (1 file + clean diff at the same upstream commit; import-verified).
Ops: one post-process wedge ate a completed run's artifacts (fzg5r1-128k —
timeout-cleanup killed the post-processor before lat.md/step_samples were
written; re-ran clean). Next in the armed sequencer: chain F2REV — the
user's reverse FSDP2 queue (70 cells, blocks 8→1, dedupe-vs-forward-agent,
pool-scoped guard); fsdp2_offload backend confirmed wired by the baseline
agent.

### §8 verdicts (as they land)

- [08-02] chain-Y 1-10/36: ALL as predicted — 122b R2 uns-OFF 192k/256k/288k/
  320k/336k = HOST-C-OOM ×5; 122b R2 rc 320k/336k + uns 336k = G-OOM ×3;
  35b R2 rc 512k/576k = G-OOM ×2. Zero surprises.
- [08-03 01:5x] cell 11 (35b R2 rc @640k) POST-MORTEM: trainer died 17:16Z —
  rank-1 allocation death → peer NCCL watchdog SIGABRT "collective operation
  timeout" = the R2A-codified **NCCL@edge G-OOM class → banked as measured
  G-OOM** (rc massively beyond its (384k,400k]-class wall; no re-run needed).
  The 8.5h stall after death: the trainer's orphaned multiprocessing spawn
  worker (reparented to init, 18 pipe fds) held tp_probe_fill's $()-capture
  open — killed by exact PID; wrapper then classified the stale capture
  HARDFAIL → chain aborted by design. **chain-Y2 launched** with the 25
  remaining cells + two hardenings: (a) probe output captured to a FILE
  (orphan writers can't wedge it), (b) 3h timeout per probe, (c) NCCL-
  timeout/SIGABRT classified G-OOM (NCCL@edge) instead of HARDFAIL.
  OPS LORE: on OOM-kills under torchrun, spawn workers can outlive the tree
  holding inherited stdout fds — never $()-capture a probe that manages its
  own children; file-capture always.

## §9 FSDP2 REVERSE QUEUE — CLOSED 2026-08-04 (user-ordered stop after
## cross-machine convergence)

Chain F2rev3 ran the user's 8-block queue in REVERSE (blocks 8→1) while the
forward baseline agent ran it forward from the SFT-39 checkout. **Stopped by
the user at cell ff30-80k** once the two tables were shown consistent — every
remaining reverse cell was already measured forward.

**Measured by this chain (c18, pool 0/1):**
- GLM-4.7-Flash R1 [3609, 2103, 1535, 1187, 955, 806] + R2 [7287, 4322, 3072,
  2375, 1934, 1608] — all 12 FIT.
- GLM-4.5-Air R1+R2 = **HOST-C-OOM ×12** (fp32 host residency ~2× the 106B
  model ≫ 957 GB; measured zero3 fits every rung — the host-efficiency
  contrast between the two "equivalent" offload stacks is the §8/§9 finding).
- 35b: 512k G-OOM, 256k FIT 1046 @b1 (65% HBM; +24% over rc's 844).
- 30b: 1.12M-r2, 720k-r2, 1.6M, 480k = G-OOM ×4 (wall from 480k confirmed).
- SO-confirm 35b@384k pair: accepted standing measured rc G-OOMs (both ranks).

**Cross-machine consistency (SFT-39 forward vs this chain):** every overlap
agrees ≤2.2% — GLM-4.7 R1 to ±2 tok/s (3609/2104/1533/1186/954/805 fwd),
GLM-4.5 identical all-host-OOM verdicts, 35b 256k 1037 vs 1046 (+0.9%),
identical 512k/480k walls. Cross-backend sanity: forward fsdp2 32b R1
1097/936 vs my zero3 1093/937 (0.4%) — fsdp2≈zero3 on dense, as expected.

**Not run by me (forward table stands, banked in DATA with fc* provenance):**
30b fits (5204@80k b4 / 3038@128k b2 / 1446@320k), 32b R1 1097/936 + R2
2178@128k (wall 168k — edge-flip vs SO's 1738 fit), llama R1 1091@96k + R2
all-OOM (edge-flip: SO fits 104k), 122b all-host-OOM both ranks, mixtral
all-host-OOM, 35b 128k@b2 1166 + the **starred non-monotonic 384k fits
(R1 1321 > 256k's 1037; R2 2547 > 2090)** — flagged to the user, shipped on
their stop order. My batch-blind DEDUPE-SKIP of ff35-128kb2 is superseded by
the forward b2 measurement (f2_catchup.list retired).

**Close-out ops:** chain sequencer + current cell + stale r2oomwatch watchers
killed by exact PID (the /proc scanner matched my own tool-shell AGAIN via the
pgrep line in the same script — third occurrence; lesson re-learned: bracket-
class patterns or separate commands). GPUs 0/1 clean; 2/3 hold ~3.7 GB dead
ghosts (no live owner, harmless). Both plot scripts had DUPLICATE
fsdp2_offload keys (mine + forward's in one dict literal, Python last-wins
silently rendering forward's) — deduped to the c18 chain values with forward
replication noted in comments; glm4.5 pairs were identical. Figures
regenerated (host python3.11 + user-site matplotlib 3.11.1 — container venvs
are py3.12-only) and pushed figures-only to Overleaf as **2596e25**.

## §4 CAMPAIGN VERDICTS (2026-07-29)

**37 runs, 0 hardfails, every placeholder resolved.** Per-figure state:
- q3.5-35b R1+R2, q3.5-122b R1+R2: 100% measured (R1 was already; R2 filled
  here). Upgrades: 35b R2 sole-coverage starts 576k (wall tightened); 122b R2
  128k column FLIPS to asym +15% (b2 squeeze 1665 vs uns b2 1445).
- q3-30b R1+R2: 100% measured; 80k column re-anchored to c18 best-batch.
- q3-32b R1+R2: 100% measured; uns-OFF wall corrected (384k,448k] (est ~600k
  was fiction — the 576k est cell died with it).
- llama3.3 R1+R2: 100% measured; uns-OFF wall corrected (192k,320k] (est ~416k
  fiction); 352k asym cell measured 349 with knife-edge note; 416k re-measured
  300 (lost-artifacts est was +0.3% exact).
- Est-methodology scorecard: 27 est cells → measured; median |error| ~2%; max
  errors: uns-OFF@128k-30b −23%, both uns-OFF est WALLS (~100-150k too deep),
  and the c14-edge-tax 80k column (+22-37%). Ratio/trend ests were excellent;
  extrapolated walls and edge cells were the systematic liars.
- FSDP2/ZeRO3 rows: intentionally derived-from-recomp (deterministic
  multipliers, est-bordered) in both scripts — NOT measured here; measuring
  them = new baseline configs, a separate user decision.
- OVERLEAF SYNC (07-29): push 8acbe14 = 11 fill PDFs + prose (six models,
  35b-R2 wall 576k, 1.75×). Then user ruled **122b belongs in BOTH paper
  figures** → COMBINED_KEYS gained q3.5-122b-a10b in both scripts (rank-1 now
  7 panels, rank-2 5 panels; 2r builder generalized to the rank-1 auto-grid;
  blank-slot panels get the x-title) → push 8aec12e (combined PDFs + "seven
  models" + 122b 672k/2.33× capacity claim). Verified post-push: remote blobs
  == fill commit for all per-model PDFs (parallel paper campaign's 5 commits
  landed between the two pushes, no clobber either direction).
- PHI-3.5-MOE REMOVED from all tp figure output (user ruling 07-29, push
  4c6c26e): strictly unsuitable — 131k sliding-window drops every system onto
  masked-SDPA past the window (~4000→1300 collapse), the HF stack's seq²
  window-mask materialization (61 GiB @256k) walls all systems identically
  (model-stack limit), and the un-optimized deep end flips the sign (asym T3
  842 vs uns-OFF 905 @224k). Measured row kept in plot_tp_vs_seq.py under an
  EXCLUDED_MODELS guard + prominent note; rank-1 combined back to even 3×2
  (six models); prose enumeration updated; no other Phi references in the tex.
- PARALLEL-CAMPAIGN NOTE: mid-fill, a concurrent tpfig campaign (c14 GB200,
  2026-07-28, agent/anchors_tmp/) ADDED two rank-1 models to plot_tp_vs_seq.py
  — **Mixtral-8x22B and Phi-3.5-MoE, both born fully measured (zero est)** —
  which explains the file's mid-campaign mtime change. No collision with the
  fill edits; the combined figure now carries 7 rank-1 panels, all
  placeholder-free.

- [03:39Z] **CHAIN-E DONE**: c18 80k re-anchor pair — uns b8 4335 (97%) · asym
  T1 b6 4983 (64%; +37% over c14's 3642, and b6 fit first-try so b8 unprobed).
  Both c14 80k cells were edge/undersold; c18 column re-anchor CONFIRMED as the
  path. chainE2 staged (4 runs, after F): uns b6 + b4, asym b8, uns-OFF b8-ff →
  final column = best measured per system, all-c18. **CHAIN-F (llama, 9 cells)
  STARTED [03:39Z]** — dataset auto-builds inline.
- [03:19Z] **CHAIN-D COMPLETE (6/6, all FIT, max est-error 5.6%).** 30b/32b R2
  DATA rows updated to fully-measured (est markers gone; comments carry cells +
  batch + est-deltas). CHAINS E→F LAUNCHED as one serial job [03:2xZ]: E = 30b
  80k re-anchor (uns b8-ff, asym T1 b6-ff) to rule on the rc-5206 anomaly; F =
  9 llama cells (weights landed, snapshot 6f6073b4).

- [01:1xZ 07-29] **CHAIN-C CLOSED (11 verdicts; 2 queued cells cancelled as moot).**
  The uns-OFF@448k HOST-C-OOM lands the 32b uns-OFF wall at (384k,448k] — the
  est-wall ~600k (and its 234@576k est cell) were fiction; 576k/640k runs
  cancelled (beyond a measured host wall, monotone), chain stopped cleanly
  between runs (GPU verified empty, 576k had not left dataset-build). 30b R1:
  6/6 measured. 32b R1: 5 measured + 1 measured wall; the 576k column DROPS
  from the 32b figure row (no system alive there except asym T3 245 — row
  keeps it; uns-OFF cell becomes OOM*). CHAIN-D LAUNCHED [01:1xZ].

- [19:4xZ] chain-C c1: **30b rc @80k = 61.5 · 5206 · 150.5 (81%) · 175 · 0.2% —
  FIT at b4** (b8/b6/b5 OOM; est was 4400). ⚠️ CONSISTENCY FLAG: 5206 is +52%
  over the row's c14-banked uns 3424 (b8 @96% edge) — rc≈uns convergence law
  says the 80k column is now cross-node inconsistent (c18 fresh-healthy vs c14
  edge-taxed). QUEUED chain-E re-anchor: uns @80k b8-first-fit + asym T1 @80k
  b6-first-fit on c18 (~30-40 min) → decide whether the 80k column re-anchors
  wholesale to c18 or the rc cell carries a node footnote. NOT applied to DATA
  until the re-anchor verdict.
- [18:40Z] **CHAIN-B COMPLETE (3/3).** 122b R2 row now 100% measured; DATA
  updated: uns@128k 1374est→1445 (b2), asym@128k 950→1665 (b2 squeeze, max-TP
  rule — beats uns +15%; b1-vs-b1 would have inverted the sign), uns-OFF@128k
  OOM*→measured C-OOM. 122b R2 verdicts now: asym wins BOTH shared columns
  (+15% @128k best-batch, +31% @288k) with capacity tie at (288k,320k].
  CHAIN-C (30b+32b R1, 13 runs) LAUNCHED [18:40Z] — 30b rc@80k b8-first-fit
  running. 30b/32b datasets will auto-build on first use (c18 never ran these
  models; ~20-40 min extra on first cells per seq).

- [14:13Z] **CHAIN-A COMPLETE — 7 fills + 1 wall probe, all clean, ±5.1% max
  est-error.** 35b R2 row now 100% measured; plot_tp_vs_seq_2r.py DATA updated
  (est markers dropped; uns-OFF 576k flipped OOM*→measured OOM; comment rewritten).
  Story upgrades: asym h2h beat at every shared R2 column (+27/+37/+44%), sole
  coverage begins 576k (was 640k). tp2r figures regenerated (in-container python3;
  out/ symlinks to env/figures/out). CHAIN-B (122b R2) LAUNCHED [17:43Z].
- [~18:5xZ] chain-B b1: **122b R2 uns @128k = 354.3 · 1445 GLOBAL · 168.9 (91%) ·
  399/rank · 0.4% — FIT at b2** (b2 first-fit; est was 1374, +5.2%). The 128k
  column bar for asym is now 1445; sEP-T1 b2 squeeze in flight decides the sign
  (b1 banked 950 @63%).
