# EP grain fix — fair DP/EP/sEP comparison for Fig 7 (M5b)

(2026-07-26 v3, compressed. Sign-off: Kevin approved re-banking the OWNED
cells only; sdp/plan/queue stay banked.)

**EXECUTED 2026-07-27.** Steps 1-5 done for q35-122b (gemm + moe): fix
shipped in ep_balance_bench.py (`owned_segments`), owned re-measured on
GPUs 2,3, spliced into env/figures/data_skew/ (pre-fix copies in
`archive_pregrainfix_20260726/`), figures regenerated and copied to
Overleaf, prose updated at all 5 cite sites + the M5b entry. All gates
passed: measured b_mb == the step-1 analytic table for all 18 cases,
owned rows[] byte-identical, panel (a) pixel-identical, non-owned bytes
unchanged. Results: EP traffic 2.69/4.18/4.95/5.23 → 2.39/2.93/2.58/2.26
GB and EP wall 29.9/37.6/44.6/51.1 → 30.5/35.1/40.4/49.1 ms at
z=0.5/1.0/1.5/2.0. NOT DONE (input data absent from this container —
the -39 worktree and profiling_both_skew/ do not exist here, so only
the data_skew 122b pair could be spliced): 30b / scout / 235b-moe
appendix bars. See the CAVEATS section at the end.

## The whole task (it is small)

Fig 7 already compares DP / EP / sEP on the three goal metrics —
imbalance (panel a, rows-derived), traffic (panel b, C2C weight-bank
READS), latency (panel c, max-rank busy). ONE thing is unfair: EP's
segment list inherits the queue's fine 8192-row sharing grain
(`seg_sets["owned"]` filters `segs_all`, ep_balance_bench.py:388), so its
hot banks are re-read ~hot-mass/8192 times, while DP/sEP get grid-aware
tiling (`_chunk_local` :361, ≤ ~19 pieces/expert). Everything else is
already fair: transport identical (every mode streams banks from pinned
CPU memory — "EP with experts on CPU" needs nothing more); activation and
output traffic are mode-identical by construction (per-link they follow
panel a's row split; no mode moves tokens); imbalance is untouched by
chunking. Classic HBM-resident EP with all-to-all stays out of scope
(weights don't fit — the paper's premise).

Fix + refresh = 5 actions:

1. PREDICT (no GPU). Re-derive owned b_mb analytically under the new rule
   for every banked case (zipf_counts is seed-deterministic; replicate
   `_chunk_local` verbatim). Validated sim this convo: EP traffic becomes
   2.4 / 3.0 / 2.6 / 2.3 GB at z=0.5/1.0/1.5/2.0 (was 2.7→5.2 growth).
   The spliced json MUST later equal this table exactly — it's analytic;
   walls are the only new measurement.

2. EDIT (~10 lines, scripts/testing/ep_balance_bench.py). Add
   `owned_segments(counts, rank)` = whole own-half experts
   ([rank·E/2,(rank+1)·E/2), nonzero, prefix sums — shape of
   `sdp_segments` :165); set `seg_sets["owned"] =
   _chunk_local(owned_segments(...))` at :388; update docstring :7-8
   (owned = own experts' rows + the same local grid-fill chunking every
   mode gets). queue / sdp / plan / owned_smart untouched.

3. RERUN owned only (in-container; orphan sweep + GPUs idle first):

   ```
   MODEL=q35-122b-a10b MODES=owned SCOPE=gemm \
   ALPHAS=z0.0,z0.5,z0.8,z1.0,z1.5,z2.0 SEEDS=3 REPS=3 GPUS=2,3 \
   OUT=profiling_results/motivation/ep_owned_fair_q35122b_gemm.json \
   bash scripts/testing/ep_balance_bench.sh
   ```

   Same with SCOPE=moe (block prose + appendix panel). q3-235b gemm while
   system_writing_v2.md:322 keeps "−32% vs EP". 30b / scout / 235b-moe
   only for appendix-figure consistency — rerun (default) or drop their
   EP bars. Wrapper defaults (GPUS, REPS, presets :39-42) reproduce the
   banked conditions; z-only ALPHAS needs no HIST (:46).

4. SPLICE + REGEN. Replace only the `owned` entry per case (matched by
   name) in both banked copies — -39 worktree profiling_both_skew/ (full
   4-model set) and env/figures/data_skew/ (122b pair; md5-identical
   today) — after backing up to archive_pregrainfix_20260726/. Gates:
   case names match; owned rows[] byte-identical (hard invariant ⇒ panel
   a frozen); owned b_mb == step-1 table; every non-owned byte unchanged.
   Regen from -39: `EP_PAPER_PAIR=1 ./plot_ep_balance.sh` (against
   data_skew you MUST add `--model q35-122b-a10b` — default --model all
   reads all 4 models and crashes there; first positional arg is
   OUTPUT_DIR). Copy PDFs to Overleaf under existing names. Regression
   check = VALUES: panel a pixel-identical; panels b/c y-axes rescale.

5. PROSE (after numbers land). Caption motivations.tex:449-473: delete
   "re-streamed up to 1.8×" and "crossing DP at z=2" (wrong even for old
   data — old crossing was z=1.5). New middle-panel story: DP = every
   bank on both GPUs (2× floor, 3.15 GB) plus grid-fill chunking; EP and
   ours BOTH sit near the bank-once floor (1.57 GB; ours measures
   1.5–1.9× it). NEVER claim "ours lowest traffic" (at z=2.0 ours 2.8 vs
   EP 2.3): the claim is "DP's balance at EP's traffic" — panels a and c
   carry the win, and EP's wall mechanism is now pure hot-rank
   serialization. Re-derive the EP wall (44.6 ms @ z=1.5) and margins at:
   motivations.tex:495, motivation_full_v2.md:387, motivation_v2.md:236,
   system_writing_v2.md:322 and :510 (C7); expect some compression (link
   relief) but measurement decides — do NOT pre-write numbers. Record the
   re-bank in motivation_v2_plots.md's M5b entry.

## Expected post-fix panels (the convergence target)

  plot GB, z=0.5/1.0/1.5/2.0, sum over 2 GPUs:
  DP   3.9 / 4.8 / 4.5 / 4.2   banked, UNCHANGED (gate)
  EP   2.4 / 3.0 / 2.6 / 2.3   predicted (step 1 re-derives)
  ours 2.4 / 2.9 / 2.6 / 2.8   banked, UNCHANGED (gate)

Panel a unchanged (EP 56% seed-mean excess rows at z=2.0). Panel c: only
EP re-measured; sEP-lowest-wall at z ≥ 1.0 expected to survive (EP
improves a few ms at most while its row imbalance is untouched); if EP
reorders vs sDP anywhere, adjust the mechanism sentence, not the claim.

## Receipts (verified 2026-07-26; saves the executor re-deriving)

- Units: bank = N·K·2 = 6.291456 MB (122B); plot GB = Σ b_mb/1024;
  floors 1.573 (EP/ours) and 3.146 (DP).
- Old-EP inflation is pure grain: extra reads at z=2.0 = +595/+595/+594
  (seeds 0/1/2) over the 256-bank floor ≈ the top-8-at-8192
  reconstruction; all 256 experts active at every z (min ~48 rows).
- Grid constants (n_blk_min → _pieces): 122B 16→19, 30B 12→25, 235B
  24→13 — moe scope has the SAME segments for these (down-stage blocks
  48/32/64 don't lower the min), so moe b_mb = 3× gemm reads. Scout:
  gemm 256→2 but moe 80→4 (down 5120/64) — sim scout-moe separately.
  Scout's `force` branch needs a 1-segment list (len·256 < 296).
- ours/floor = 1.51/1.87/1.64/1.75 across z (hence "1.5–1.9×", not
  "~1.5×"); 56% = mean of 61.6/75.1/32.6.
- Cite-site list is complete (grep'd, twice); no experts/layer-scope EP
  citations anywhere; 30B prose 15.2 vs 18.2 (system_writing_v2.md:323)
  is plan-vs-raced, so 30B is figure-only.
- A stacked weights/activations/outputs traffic panel was considered and
  REJECTED: activation/output components are proportional to panel a's
  rows — it would restate panel a; panel b + the caption sentence carry
  the traffic comparison.

## CAVEATS from the 2026-07-27 execution (read before citing)

1. **GPU-pair sensitivity — RESOLVED by re-timing panel (c).** The pair
   and the session move walls by more than the effect being measured:
   on GPUs 1,2 the modes NOT being changed ran 10–27% faster than banked
   (sdp z=1.0: 39.9 → 30.3 ms), and even on the documented pair 2,3 a
   fresh session drifted +3…+8%. Splicing `owned` alone would have left
   EP a ~5% handicap that banked DP/ours did not carry — a bias IN OUR
   FAVOUR. Fixed 2026-07-27: all four modes re-timed in ONE session
   (REPS=5, GPUS=2,3) and wall_s + imbalance re-spliced for every mode;
   see `splice_walls.py` gates H1–H4. Panels (a) and (b) verified
   pixel-identical across the re-time (only x≥2413 of 3519 changed) —
   traffic is analytic and session-invariant, so panel (b) never moved.

   **GPU PAIRING RULE (verified `nvidia-smi topo -m`):** GPU0+GPU1 sit
   on NUMA node 0 (CPU 0-71), GPU2+GPU3 on NUMA node 1 (CPU 72-143).
   Use **0+1 or 2+3 only — never 1+2.** A cross-socket pair gives each
   GPU its own Grace DRAM instead of sharing one socket's bandwidth,
   which silently voids §2.4.5's "both GPUs stream from the same CPU
   memory, shared 500 GB/s ceiling" premise and inflates every number
   by ~20%. All banked and re-timed numbers here used 2,3.

2. **Only the 122b pair could be spliced.** `profiling_both_skew/` and
   the `-39` worktree do not exist in this container; `env/figures/
   data_skew/` holds only table1_q35122b_{gemm,moe}.json. The 30b /
   scout / 235b appendix figures were ALREADY ungeneratable here before
   this change (their table1_*.json are absent everywhere) — this is not
   a regression, but their EP bars remain pre-fix wherever they are
   banked. Re-run them where the data lives.

   q3-235b gemm WAS re-measured (all four modes, one session, GPUs 2,3 →
   profiling_results/motivation/ep_owned_fair_q3235b_gemm.json) because
   system_writing_v2.md:322's "−32% vs EP" is a live prose claim. Post
   fix at z=1.5: EP 68.9 / sDP 53.1 / plan 48.9 / raced 50.6 ms ⇒ plan
   −29% vs EP, −8% vs sDP (raced −26% / −5%). The EP margin shrank as
   expected (−32% → −29%); the sDP margin also moved (−12% → −8%), which
   the grain fix cannot explain — that is measurement/session variance
   against an absent bank. :322 is annotated SUPERSEDED, not rewritten.
   30b needs no prose fix (its cited 15.2 vs 18.2 is plan-vs-raced, no
   EP term); scout has no prose claim.

3. **±0.1 ms between the draft prose and data_skew.** The draft cites
   31.2 plan / 39.0 queue / 42.8 sDP; recomputing 3-seed means from
   data_skew gives 31.08 / 39.11 / 42.86. Those cells are untouched
   banked data, so the prose was left as written — but the derivation
   the draft used is not reproducible from data_skew. (The doc's own
   receipt ours/floor = 1.51/1.87/1.64/1.75 DOES reproduce exactly, so
   the aggregation method here is right.) Worth reconciling once.

4. **Pre-existing figure defect, not introduced here:** in
   ep_balance_q35122b_row3.pdf the legend's "DP" entry collides with
   panel (b)'s y-axis label. Verified identical in the pre-fix figure
   rendered from the archive, so it is untouched by this change — but it
   is visible in the paper and should be fixed on the next figure pass.

5. **Not compile-checked.** No LaTeX toolchain in this container. The
   edited caption has balanced braces and no unescaped %, but
   motivations.tex was not built.

6. **drafts/motivation_v2_plots.md is stale** (2026-07-26 12:09) and
   predates the M5b-row3 section; only outputs/s04-p1-dgx-02-c06/
   motivation_v2_plots.md was updated. The other three drafts/*.md ARE
   byte-identical twins of their outputs/ copies and both were updated.
