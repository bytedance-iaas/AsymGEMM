# plot_ep — TASK PROMPT: the EP-effectiveness bar figures (one per model)

You are working in /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM (user alias
/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM). Produce FOUR publication
figures — one per MoE model — each proving our EP balancing works, matching
the repo's existing figure style EXACTLY. Do not stop until all four render
correctly, each has passed the visual checklist below at least twice, and the
four are style-identical to each other and to the existing figures.

## WHAT TO PLOT (same layout for every model)

Per model: ONE figure, TWO side-by-side panels (separate y-scales) sharing a
single legend:
  left panel = "Expert GEMM"   right panel = "MoE Block"
X axis: four clusters per panel, one per skew setting: z0.5, z1.0, z1.5, z2.0.
Each cluster has FOUR bars, in this order and with these legend labels:
  1. "EP"          (data key: owned)   — classical owned expert-parallelism
  2. "sDP"         (data key: sdp)     — shared-bank streaming data parallel
  3. "sEP (plan)"  (data key: plan)    — ours, count-computed cut
  4. "sEP (queue)" (data key: queue)   — ours, work-stealing queue
Y axis: wall time in ms (mean over the 3 seeded shuffles). Add a thin error
cap to each bar spanning min..max over the 3 seeds. Optionally print the
GPU-imbalance % (mean) as a small in-bar or above-bar label on the EP bars
only IF it stays uncluttered — drop it if it crowds (visual check decides).

## DATA SOURCES (already measured — do NOT re-run benchmarks)

All under profiling_both_skew/. Use these top-level files ONLY —
archive_3mode/ and archive_4mode_noDP/ hold stale same-named copies.

  model key      display name (figure label)   GEMM json               MoE json
  q3-30b-a3b     Qwen3-30B-A3B                 table1_micro.json       table1c_moe.json
  q3-235b-a22b   Qwen3-235B-A22B               table1_q3235b_gemm.json table1_q3235b_moe.json
  q35-122b-a10b  Qwen3.5-122B-A10B             table1_q35122b_gemm.json table1_q35122b_moe.json
  l4-scout       Llama-4-Scout                 table1_l4scout_gemm.json table1_l4scout_moe.json

Parse recipe (all eight files share the schema, verified 2026-07-13):
  data["cases"] is a list of dicts (20 for q3-30b, 18 for the others). Each is
    {"case": str, "m_per_expert": int, "owned": {...}, "sdp": {...},
     "plan": {...}, "queue": {...}}
  GOTCHA: "m_per_expert" is an int — do NOT treat every non-"case" key as a
  mode dict; select the four mode keys explicitly.
  Each mode dict has fields: wall_s (seconds -> plot ms = wall_s*1e3),
  imbalance (0..1), b_mb, experts, rows — only wall_s and imbalance are used.
  For each z in {0.5, 1.0, 1.5, 2.0}: select the 3 cases named exactly
  f"zipf{z}|seed{n}" (n = 0..2), aggregate mean (bar height) and min/max
  (error caps). Ignore "zipf0.0", "zipf0.8", "worst:...", "median:..." cases.

## STYLE — MUST MATCH THE EXISTING FIGURES

  - Import and use scripts/figures/constants.py (module `constants`) for
    EVERYTHING stylable: font family setup, FONT_SIZE_BASE/TICK/LEGEND/
    AXIS_TITLE/SEGMENT, GRID_COLOR, SPINE_COLOR, and the BAR_* palette.
    Do NOT hardcode any color/fontsize that constants.py already defines.
    Sizing/spacing knobs go in a NEW constants.FIGURE_PARAMS["ep_balance"]
    entry (copy the "main" entry's shape), not inline in the script.
  - Bar colors (from the BAR palette, consistent with the repo's semantics of
    "muted = baselines, navy hero + saturated = ours"):
      EP -> BAR_NEUTRAL, sDP -> BAR_NEUTRAL_CYAN,
      sEP (plan) -> BAR_NAVY, sEP (queue) -> BAR_TEAL.
    White bar edges, like the other grouped-bar figures.
  - Read scripts/figures/plot_main.py FIRST and mirror its structure:
    constants.grouped_layout for the 4-bar clusters, xlim_from_positions +
    figure_size_for_xlim for sizing (sum the two panels' widths; height from
    FIGURE_PARAMS), y-grid only (GRID_COLOR, dashed, behind bars), full box
    frame (SPINE_COLOR) on BOTH panels, NO figure title, horizontal_legend on
    top, savefig dpi from the params entry, output BOTH .pdf and .png.
  - X tick labels: "z=0.5", "z=1.0", "z=1.5", "z=2.0" (FONT_SIZE_TICK); panel
    captions "Expert GEMM" / "MoE Block" as under-axis labels (FONT_SIZE_MODEL,
    bold), the way plot_main labels model groups — not axes titles.
  - Y axis label "Wall time (ms)" (FONT_SIZE_AXIS_TITLE) on each panel (the
    two y-scales differ ~5-10x, so both panels are labeled).
  - The MODEL is identified by the output filename and (if it fits the
    plot_main pattern cleanly) a small under-axis display-name label — never a
    figure title. The four figures must be layout-identical: only bar heights,
    y-scales, and the model label differ.
  - New script: scripts/figures/plot_ep_balance.py with --model
    {q3-30b-a3b,q3-235b-a22b,q35-122b-a10b,l4-scout,all}, default all; plus a
    plot_ep_balance.sh wrapper copying the pattern of plot_main.sh.
  - Outputs (stems match the json naming): scripts/figures/out/
      ep_balance_q330b.{pdf,png}    ep_balance_q3235b.{pdf,png}
      ep_balance_q35122b.{pdf,png}  ep_balance_l4scout.{pdf,png}

## ITERATION PROTOCOL (do not skip)

  1. Read constants.py and plot_main.py fully before writing code.
  2. Write the script; run it via the .sh wrapper; it must exit 0 and emit
     all eight files.
  3. VISUAL CHECK — open/Read every rendered PNG and verify EVERY item:
     [ ] four clusters x four bars per panel, correct order and colors
     [ ] bar heights match the ANSWER KEY (spot-check ≥2 values per model)
     [ ] error caps visible but subtle; no bar clipped by the axis
     [ ] fonts identical family/sizes to memory_saving_*.png (compare
         side-by-side by opening both)
     [ ] legend: ONE shared legend per figure, 4 entries, swatches match
         bar fills, no overlap with bars
     [ ] y-grid behind bars; full box frame both panels; no title; no stray
         text
     [ ] imbalance labels (if kept) legible and non-overlapping
     [ ] the four figures are pixel-consistent in style (open two at once):
         same legend position, same fonts, same bar geometry — only the data,
         y-scales, and model label differ
  4. Fix anything that fails and re-render; repeat until the checklist passes
     twice in a row on ALL FOUR figures. Log each iteration (what failed ->
     what changed) at the bottom of THIS file under "## RUN LOG".
  5. Final: verify each PDF also renders (pdftoppm or equivalent spot check).

## ANSWER KEY (means in ms over the 3 seeds, parsed 2026-07-13 from the files
## above; your parsed means must match to ±0.1 ms — this doubles as the
## checklist spot-check. Last column = EP (owned) mean GPU-imbalance %.)

  q3-30b-a3b (Qwen3-30B-A3B)
    GEMM z0.5: EP  15.7  sDP 14.5  plan 15.4  queue 15.4   imb  9%
         z1.0: EP  18.7  sDP 15.2  plan 15.4  queue 15.9   imb 39%
         z1.5: EP  22.3  sDP 15.4  plan 15.4  queue 16.2   imb 66%
         z2.0: EP  24.9  sDP 16.2  plan 15.2  queue 18.2   imb 81%
    MoE  z0.5: EP  91.0  sDP 88.0  plan 87.7  queue 86.4   imb  9%
         z1.0: EP 103.1  sDP 91.0  plan 87.8  queue 88.0   imb 30%
         z1.5: EP 119.6  sDP 94.5  plan 87.8  queue 91.7   imb 52%
         z2.0: EP 130.8  sDP 96.9  plan 88.7  queue 96.9   imb 65%
  q3-235b-a22b (Qwen3-235B-A22B)
    GEMM z0.5: EP  47.1  sDP  44.6  plan  44.5  queue  43.3   imb 13%
         z1.0: EP  57.6  sDP  48.4  plan  45.2  queue  45.4   imb 41%
         z1.5: EP  70.8  sDP  54.8  plan  48.3  queue  51.6   imb 62%
         z2.0: EP  79.9  sDP  55.2  plan  56.1  queue  56.6   imb 78%
    MoE  z0.5: EP 201.9  sDP 196.3  plan 190.1  queue 189.8   imb 11%
         z1.0: EP 242.5  sDP 208.3  plan 191.9  queue 194.8   imb 35%
         z1.5: EP 286.7  sDP 223.6  plan 196.8  queue 215.3   imb 55%
         z2.0: EP 316.1  sDP 227.5  plan 209.5  queue 230.1   imb 69%
  q35-122b-a10b (Qwen3.5-122B-A10B)
    GEMM z0.5: EP  29.9  sDP  32.3  plan  29.6  queue  29.7   imb  6%
         z1.0: EP  37.6  sDP  39.9  plan  30.7  queue  32.6   imb 19%
         z1.5: EP  44.6  sDP  42.9  plan  31.1  queue  39.1   imb 34%
         z2.0: EP  51.1  sDP  42.0  plan  36.1  queue  42.9   imb 56%
    MoE  z0.5: EP 156.7  sDP 166.2  plan 153.8  queue 151.9   imb  5%
         z1.0: EP 179.1  sDP 189.2  plan 158.8  queue 160.2   imb 15%
         z1.5: EP 206.4  sDP 201.5  plan 161.9  queue 179.8   imb 31%
         z2.0: EP 228.9  sDP 194.8  plan 171.7  queue 191.6   imb 47%
  l4-scout (Llama-4-Scout)
    GEMM z0.5: EP 220.5  sDP 198.9  plan 198.7  queue 199.1   imb 22%
         z1.0: EP 255.8  sDP 199.0  plan 200.4  queue 198.5   imb 46%
         z1.5: EP 294.5  sDP 198.3  plan 208.8  queue 200.3   imb 66%
         z2.0: EP 328.7  sDP 202.5  plan 210.6  queue 201.6   imb 79%
    MoE  z0.5: EP 527.6  sDP 492.3  plan 490.3  queue 491.6   imb 16%
         z1.0: EP 582.6  sDP 491.0  plan 490.4  queue 489.3   imb 34%
         z1.5: EP 644.6  sDP 493.0  plan 500.0  queue 490.4   imb 50%
         z2.0: EP 706.1  sDP 498.8  plan 506.2  queue 494.2   imb 62%

  The visual story every figure must convey at a glance: EP's bars climb
  steeply with skew while ours (navy plan / teal queue) stay near-flat and
  lowest-or-tied. sDP varies by model — flat for q3-30b and l4-scout, climbing
  for q3-235b, and ABOVE EP at low skew for q35-122b (shared-expert cost) —
  which is exactly why the per-model figures matter: sEP wins everywhere,
  sDP does not. If your parsed numbers break this story, re-check the parse
  before touching the data.

## RUN LOG (append-only)

- 2026-07-13 iter 1: installed fonts-urw-base35 (Nimbus Sans was missing ->
  would have silently fallen back to DejaVu and failed the font check);
  added FIGURE_PARAMS["ep_balance"] to constants.py; wrote plot_ep_balance.py
  + .sh. Rendered 8/8 files, exit 0. Parsed means == answer key (two 0.1 ms
  boundary-rounding diffs: q3-30b MoE z0.5 queue 86.5 vs 86.4, q3-235b GEMM
  z0.5 queue 43.2 vs 43.3 — within the stated ±0.1). Imbalance labels kept
  (13pt gray above EP bars — uncluttered). FAIL: left "Wall time (ms)" y-label
  clipped at the figure edge on l4scout (3-digit y ticks eat the margin).
- 2026-07-13 iter 2: constants ep_balance side_left_in 0.95->1.15,
  mid_extra_in 0.35->0.55. Re-rendered; checklist PASS #1 on all four figures
  (visual, item-by-item; edge-crop zoom confirmed no stray text; fonts
  compared side-by-side vs memory_saving_hbm.png — same face/sizes).
- 2026-07-13 pass #2 (independent re-verification): pixel-sampled all four
  PNGs — exact BAR_NEUTRAL/BAR_NEUTRAL_CYAN/BAR_NAVY/BAR_TEAL hexes present in
  bulk; all PNGs identical 3652x1200 (layout pixel-consistent across models);
  all four PDFs render via pdftoppm and match the PNG content. Checklist has
  passed twice in a row -> DONE. Also appended plot_ep_balance.sh to plot.sh.

### 2026-07-13 addendum: per-bar reduction % labels (user request)

- memory_saving_hbm: annotation changed from ours-only to CHAINED arrows —
  every bar after the first shows red bold arrow-% vs the bar to its LEFT
  (Act.Offload vs Recompute: 47/39/31/35/36 percent down; Ours vs Act.Offload:
  unchanged 22/13/31/23/21). Sign-aware (up-arrow if a bar regresses). DRAM
  figure deliberately untouched (annotate_saving stays False — it is the cost
  figure). All five chain values hand-verified against MODELS data.
- ep_balance figures: every non-EP bar gets reduction labels; colors = the
  reference (in-panel key, left panel only): red = vs EP, dark blue = vs sDP.
  sDP bar: one red label. sEP plan/queue: two label columns (red left, blue
  right, +-0.010 data-unit offset). EP keeps its gray imbalance %. "0%" when
  |delta| < 0.5%; sign-aware arrows (e.g. q35 GEMM z2.0 queue: dn16% vs EP,
  up2% vs sDP — both verified by hand from the means).
- iter 3 (horizontal 2-line stacks) FAILED visual check: adjacent stacks merge
  into one blob when plan/queue bar tops are equal (all MoE panels).
- iter 4: rotated all annotation text 90 deg (incl. the gray imbalance %),
  y-headroom 1.15 -> 1.20, cap-to-label pad 0.014. Rotation makes each label
  column ~1 glyph wide -> no collisions at any bar height. PASS #1: all four
  read cleanly incl. worst case (q3-235b GEMM z2.0: three near-equal bars,
  five distinct columns legible). PASS #2: re-render + pixel check (bar hexes
  exact-match in bulk; ACCENT_RED/DARK_BLUE present — exact-match only in the
  horizontal key since rotated glyphs are fully anti-aliased, as expected);
  all PDFs re-render via pdftoppm and match; DRAM png confirmed unchanged.
