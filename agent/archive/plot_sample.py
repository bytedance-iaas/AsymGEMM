# """Plot per-step component memory breakdown for FSDP (baseline) vs Virtu.

# Reads `memory_breakdown_rank{0..N}.jsonl` from `<profile-dir>/fsdp/` (or
# `<profile-dir>/baseline/`) and `<profile-dir>/virtu/`, snapshots each step at
# phase=`before_optimizer_step`, then renders each panel as a *stacked band
# chart* with 8 component bands per panel (in a fixed order shared across
# panels):

#     PLE     | Weight, Gradient, Optimizer State, Activation
#     Others  | Weight, Gradient, Optimizer State, Activation

# PLE = `ple_table` + `ple_dense_side`.
# Others = every other component (mlp, attention_full/sliding, norms, other_params,
# token_embedding_lm_head_shared, vision_projector, audio).

# Activation accounting:
# - `PLE — Activation` = `ple_activation_bytes` field, populated by forward
#   pre/post hooks installed by `MemoryBreakdownProfiler` on PLE submodules
#   (sum of `memory_allocated()` deltas around their forward calls during the
#   step's first forward pass).
# - `Others — Activation` = (`peak_allocated_since_step_begin` − persistent
#   weight+grad+opt across all components) − `PLE — Activation`. This is a
#   residual that captures the rest of the within-step activation watermark.
# - The 8 lines therefore sum to the peak (within-step) allocated bytes — the
#   OOM-relevant memory pressure, not the steady-state snapshot.

# Renders 4 panels (2x2):
#     [FSDP Max Peak]   [FSDP Avg Peak]
#     [Virtu Max Peak]  [Virtu Avg Peak]

# where Max/Avg are taken across ranks at the chosen phase using
# `peak_allocated_since_step_begin`.

# Color palette is sampled from `agent/overleaf/[NeurIPS'26] Efficient PLE/figs/method.pdf`:
# warm hues (peach -> red) for PLE, cool hues (sage -> navy) for Others.

# Usage:
# python scripts/plots/plot_memory_breakdown.py --profile-dir output/memory_profile/model-gemma-4-E4B-it_data-dolma3_50k.json_steps-10_lr-1e-5_bs-4_pad-max-batch_msl-4096_gpus-4_ga-1_wd-0.0_seed-42
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# from collections import defaultdict
# from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib import font_manager
# from matplotlib.patches import Patch
# import numpy as np
# import scienceplots  # noqa: F401

# plt.style.use(["science", "no-latex"])


# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# PHASE = "before_optimizer_step"
# BYTES_PER_GB = 1024 ** 3  # GiB; labelled as "GB" to match common ML usage

# PLE_COMPONENTS = {"ple_table", "ple_dense_side"}
# GROUPS = ("PLE", "Others")
# CATEGORIES = ("Weight", "Gradient", "Optimizer", "Activation")
# KIND_KEY = {"Weight": "weight", "Gradient": "grad", "Optimizer": "optimizer_state"}

# # =============================================================================
# # USER-TUNABLE: per-band fill colours (hex). Edit any row to recolour that band
# # in every panel; the legend swatch follows automatically. Order is irrelevant
# # here — the bottom-to-top stacking order is set separately by STACK_ORDER.
# # Defaults sampled from agent/samples/Screenshot 2026-05-06 at 8.02.55 PM.png.
# # =============================================================================
# # PALETTE: dict[tuple[str, str], str] = {
# #     ("PLE",    "Weight"):     "#93C18E",  # sage green   (screenshot #8, bottom band)
# #     ("PLE",    "Gradient"):   "#F1D9AE",  # cream peach  (screenshot #7)
# #     ("PLE",    "Optimizer"):  "#A93232",  # deep red     (screenshot #6)
# #     ("PLE",    "Activation"): "#EFA9A8",  # salmon pink  (screenshot #5)
# #     ("Others", "Weight"):     "#6FACD7",  # sky blue     (screenshot #4)
# #     ("Others", "Gradient"):   "#6E3D2A",  # dark brown   (screenshot #3)
# #     ("Others", "Optimizer"):  "#DDDDDD",  # light gray   (screenshot #2)
# #     ("Others", "Activation"): "#B89BCF",  # lavender
# #     # FSDP-overheads band: peak − every directly measured component above.
# #     # Captures FSDP all-gather scratch (transient unsharded params), optimizer
# #     # step temporaries, and allocator caching. This is the slack Virtu
# #     # eliminates by keeping the PLE table out of FSDP's flat-buffer.
# #     ("FSDP",   "Overheads"):  "#F2A8B0",  # salmon (screenshot #1, top band)
# # }
# PALETTE: dict[tuple[str, str], str] = {
#     ("PLE",    "Weight"):     "#F3C48E",  # method-draft green row tiles
#     ("PLE",    "Gradient"):   "#337296",  # method-draft orange key/gradient boxes
#     ("PLE",    "Optimizer"):  "#9D281E",  # method-draft purple optimizer states
#     ("PLE",    "Activation"): "#7C8354",  # method-draft yellow tau/state boxes
#     ("Others", "Weight"):     "#BFDBE8",  # method-draft pale blue lookup/transfer tint
#     ("Others", "Gradient"):   "#F1F1F1",  # method-draft blue communication arrows
    
#     ("Others", "Optimizer"):  "#FFD59A",  # method-draft inactive/neutral rows
#     ("Others", "Activation"): "#D7EEF4",  # method-draft green checks/active flow
#     # FSDP-overheads band: peak − every directly measured component above.
#     # Captures FSDP all-gather scratch (transient unsharded params), optimizer
#     # step temporaries, and allocator caching. This is the slack Virtu
#     # eliminates by keeping the PLE table out of FSDP's flat-buffer.
#     ("FSDP",   "Overheads"):  "#949F75",  # method-draft red hot/congestion highlight
# }

# # =============================================================================
# # USER-TUNABLE: font sizes (in points). Bump any value to enlarge that element
# # across all panels; matplotlib resolves these at draw time, so re-running the
# # script with edited numbers is enough.
# # =============================================================================
# def configured_sans_serif_stack() -> list[str]:
#     """Register optional Helvetica Neue font files and return fallback stack."""
#     font_paths: list[Path] = []
#     font_path_env = os.environ.get("HELVETICA_NEUE_FONT_PATH")
#     if font_path_env:
#         font_paths.extend(Path(item).expanduser() for item in font_path_env.split(os.pathsep) if item)

#     font_dir_env = os.environ.get("HELVETICA_NEUE_FONT_DIR")
#     if font_dir_env:
#         font_paths.extend(
#             Path(path)
#             for path in font_manager.findSystemFonts(
#                 fontpaths=[str(Path(font_dir_env).expanduser())],
#                 fontext="ttf",
#             )
#         )

#     families: list[str] = []
#     for font_path in font_paths:
#         if not font_path.exists():
#             raise SystemExit(f"Configured Helvetica Neue font does not exist: {font_path}")
#         font_manager.fontManager.addfont(str(font_path))
#         families.append(font_manager.FontProperties(fname=str(font_path)).get_name())

#     fallback = [
#         "Helvetica Neue",
#         "Neue Helvetica",
#         "Helvetica Neue LT Pro",
#         "Helvetica",
#         "Nimbus Sans",
#         "Arial",
#         "DejaVu Sans",
#     ]
#     return list(dict.fromkeys(families + fallback))


# FONT_FAMILY          = "sans-serif"
# FONT_SANS_SERIF      = configured_sans_serif_stack()
# MATH_TEXT_FONTSET    = "dejavusans"
# TITLE_FONTSIZE       = 30    # per-panel title (e.g. "FSDP — Max Peak")
# TITLE_PAD            = 14    # vertical gap between each panel title and axes
# XLABEL_FONTSIZE      = 30    # x-axis label ("Step")
# YLABEL_FONTSIZE      = 30    # y-axis label ("Memory (GB)")
# XTICK_FONTSIZE       = 28    # x-axis tick numerals
# YTICK_FONTSIZE       = 28    # y-axis tick numerals
# Y_TICK_COUNT         = 3     # linspace ticks from y-min to y-max; 3 -> 0, mid, top
# Y_TICK_LABEL_FORMAT  = "{value:.0f}"
# LEGEND_FONTSIZE      = 26    # legend text
# LEGEND_HANDLE_LENGTH = 1.5   # legend swatch width  (matplotlib units)
# LEGEND_HANDLE_HEIGHT = 1.2   # legend swatch height (matplotlib units)
# LEGEND_COLUMN_SPACING = 1.4  # horizontal gap between legend entries
# LEGEND_X_ANCHOR      = 0.055 # left edge of the shared legend in figure coords
# LEGEND_Y_ANCHOR      = 0.99  # top edge of the shared legend in figure coords
# LEGEND_ITEMS_PER_ROW_BY_LAYOUT = {
#     "stacked": 3,      # 3 columns -> 3 rows for the 9 memory bands
#     "horizontal": 5,   # 5 columns -> 2 rows for the 9 memory bands
# }
# LEGEND_TOP_RESERVE_BY_LAYOUT = {
#     "stacked": 0.23,    # fraction of figure height reserved above panels
#     "horizontal": 0.30, # for the legend in each layout
# }
# STACKED_FIG_WIDTH    = 14.0
# STACKED_ROW_HEIGHT   = 4.7
# STACKED_MIN_HEIGHT   = 7.0

# # =============================================================================
# # USER-TUNABLE: which per-step views to render. Each entry becomes a row in
# # the figure (FSDP in the left column, Virtu in the right). Valid options:
# #   "maxrank" — breakdown of the rank with the largest peak that step
# #   "avgrank" — per-component mean across ranks
# #   "rank0"   — rank 0's own breakdown (single rank, no aggregation)
# # Order matters: rows are rendered top-to-bottom in the order listed.
# # Examples:
# #   VIEWS_TO_RENDER = ("maxrank", "avgrank")              # 2 rows, 4 panels
# #   VIEWS_TO_RENDER = ("rank0",)                          # 1 row, 2 panels
# #   VIEWS_TO_RENDER = ("maxrank", "avgrank", "rank0")     # 3 rows, 6 panels
# # =============================================================================
# # VIEWS_TO_RENDER: tuple[str, ...] = ("rank0",) # ("maxrank", "avgrank", "rank0")
# VIEWS_TO_RENDER: tuple[str, ...] = ("rank0", "avgrank") # ("maxrank", "avgrank", "rank0")

# # Figure panel layout:
# #   "stacked"    -> one row per view, two columns: FSDP | Virtu
# #   "horizontal" -> one row with all panels left-to-right: FSDP, Virtu, ...
# PANEL_LAYOUT = "stacked"

# _VIEW_TITLES = {
#     "maxrank": ("FSDP — Max Peak (across ranks)", "Virtu — Max Peak (across ranks)"),
#     "avgrank": ("FSDP — Avg across Ranks", "Virtu — Avg across Ranks"),
#     "rank0":   ("FSDP — Rank 0",                  "Virtu — Rank 0"),
# }

# # Panel-level accent: matches method.pdf's "problem (red) vs solution (green)" framing.
# BACKEND_ACCENT = {"FSDP": "#8C2A2A", "Virtu": "#3F7F5A"}

# # Canonical bottom-to-top stack order. The same order is used in every panel
# # (PLE bands at the bottom, Others on top; W -> G -> Opt -> Act within each
# # group), so visually identical y-positions across panels mean identical
# # components.
# STACK_ORDER: tuple[tuple[str, str], ...] = (
#     ("PLE", "Weight"),
#     ("PLE", "Gradient"),
#     ("PLE", "Optimizer"),
#     ("PLE", "Activation"),
#     ("Others", "Weight"),
#     ("Others", "Gradient"),
#     ("Others", "Optimizer"),
#     ("Others", "Activation"),
#     ("FSDP", "Overheads"),  # 9th band — FSDP scratch + opt temps + allocator misc
# )

# # Top legend order, independent of the bottom-to-top stack order. This is the
# # desired visual row-major order; _legend_column_major_input_order converts it
# # for matplotlib's column-major legend packing.
# LEGEND_ORDER: tuple[tuple[str, str], ...] = (
#     ("Others", "Weight"),
#     ("Others", "Gradient"),
#     ("Others", "Optimizer"),
#     ("Others", "Activation"),
#     ("FSDP", "Overheads"),
#     ("PLE", "Weight"),
#     ("PLE", "Gradient"),
#     ("PLE", "Optimizer"),
#     ("PLE", "Activation"),
# )


# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------

# def load_phase_records(profile_dir: Path, phase: str = PHASE) -> list[dict]:
#     """Return detailed rank-jsonl records matching `phase`, across all ranks."""
#     records: list[dict] = []
#     files = sorted(profile_dir.glob("memory_breakdown_rank*.jsonl"))
#     if not files:
#         raise FileNotFoundError(
#             f"No memory_breakdown_rank*.jsonl files under {profile_dir}"
#         )
#     for jsonl in files:
#         with jsonl.open() as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 rec = json.loads(line)
#                 if rec["phase"] == phase and rec["detailed"] is True:
#                     records.append(rec)
#     if not records:
#         raise ValueError(f"No detailed records with phase={phase!r} in {profile_dir}")
#     return records


# def breakdown_for_record(rec: dict) -> dict[tuple[str, str], int]:
#     """Return {(group, category): bytes} for one (rank, step) record.

#     Now produces 9 bands:
#     - 6 directly measured persistent bands (PLE/Others × Weight/Grad/Opt) from
#       `component_bytes`.
#     - PLE-Activation from the profiler's PLE forward-hook accumulator
#       (`ple_activation_bytes`).
#     - Others-Activation from the profiler's non-PLE forward-hook accumulator
#       (`others_activation_bytes`).
#     - Misc-Residual = `peak_allocated_since_step_begin` minus all of the above
#       — captures FSDP all-gather scratch + optimizer-step temps + allocator
#       misc that aren't directly attributable.

#     Bands sum to the within-step peak.
#     """
#     out: dict[tuple[str, str], int] = {(g, c): 0 for g in GROUPS for c in CATEGORIES}
#     out[("FSDP", "Overheads")] = 0

#     persistent = 0
#     comps = rec["component_bytes"]
#     for comp_name, kinds in comps.items():
#         group = "PLE" if comp_name in PLE_COMPONENTS else "Others"
#         for cat in ("Weight", "Gradient", "Optimizer"):
#             v = int(kinds[KIND_KEY[cat]])
#             out[(group, cat)] += v
#             persistent += v

#     ple_activation = max(0, int(rec["ple_activation_bytes"] or 0))
#     others_activation = max(0, int(rec["others_activation_bytes"] or 0))
#     peak = int(rec["peak_allocated_since_step_begin"])
#     residual = max(0, peak - persistent - ple_activation - others_activation)

#     out[("PLE", "Activation")] = ple_activation
#     out[("Others", "Activation")] = others_activation
#     out[("FSDP", "Overheads")] = residual
#     return out


# def per_rank_lines(records: list[dict], rank: int = 0) -> tuple[list[int], dict]:
#     """Per-step breakdown for a single rank only.

#     Returns (steps, lines) where lines maps (group, category) -> list of
#     bytes aligned with `steps`, taken from the chosen rank's records.
#     """
#     by_step: dict[int, dict] = {}
#     for rec in records:
#         if int(rec["rank"]) != rank:
#             continue
#         by_step[int(rec["step"])] = breakdown_for_record(rec)

#     steps = sorted(by_step)
#     keys = list(STACK_ORDER)
#     lines = {k: [float(by_step[s][k]) for s in steps] for k in keys}
#     return steps, lines


# def per_step_lines(records: list[dict]) -> tuple[list[int], dict, dict]:
#     """Aggregate records into max-rank and avg-across-ranks per step.

#     Returns (steps, max_lines, avg_lines) where each *_lines dict maps
#     (group, category) -> list[float bytes] aligned with `steps`.

#     Definitions (matching `memory_step_summary.md` semantics on the totals):
#     - Max: pick the rank with the largest `peak_allocated_since_step_begin` for that step,
#       then take its per-component breakdown. Lines sum to the max-rank
#       peak, so the decomposition is physically realisable on a single GPU.
#       (Naively taking per-component max-across-ranks would over-count, since
#       different components can peak on different ranks.)
#     - Avg: per-component mean across ranks. Lines sum to mean peak.
#     """
#     by_step_rank: dict[int, dict[int, tuple[dict, int]]] = defaultdict(dict)
#     for rec in records:
#         step = int(rec["step"])
#         rank = int(rec["rank"])
#         by_step_rank[step][rank] = (
#             breakdown_for_record(rec),
#             int(rec["peak_allocated_since_step_begin"]),
#         )

#     steps = sorted(by_step_rank)
#     keys = list(STACK_ORDER)
#     max_lines = {k: [] for k in keys}
#     avg_lines = {k: [] for k in keys}
#     for step in steps:
#         per_rank = list(by_step_rank[step].values())
#         # Avg: per-component mean across ranks.
#         for k in keys:
#             avg_lines[k].append(float(np.mean([b[k] for b, _ in per_rank])))
#         # Max: breakdown of the rank with the highest peak allocation.
#         max_breakdown, _ = max(per_rank, key=lambda br_alloc: br_alloc[1])
#         for k in keys:
#             max_lines[k].append(float(max_breakdown[k]))
#     return steps, max_lines, avg_lines


# # ---------------------------------------------------------------------------
# # Plotting
# # ---------------------------------------------------------------------------

# def plot_panel(
#     ax,
#     steps,
#     lines,
#     title,
#     *,
#     show_xlabel=True,
#     show_ylabel=True,
#     show_xticklabels=True,
# ):
#     """Stacked area panel: 8 colored bands stacked bottom-to-top in STACK_ORDER.

#     Title is rendered as a plain panel title above the axes (no banner / no
#     italic / no bold) to match the requested aesthetic. Y-axis is memory in
#     GB; band heights at each step sum to the within-step peak.
#     """
#     band_data = np.asarray(
#         [np.asarray(lines[k]) / BYTES_PER_GB for k in STACK_ORDER]
#     )
#     colors = [PALETTE[k] for k in STACK_ORDER]
#     labels = [_band_label(k) for k in STACK_ORDER]

#     ax.stackplot(
#         steps,
#         band_data,
#         colors=colors,
#         labels=labels,
#         edgecolor="white",
#         linewidth=0.4,
#         antialiased=True,
#     )

#     ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
#     if show_xlabel:
#         ax.set_xlabel("Step", fontsize=XLABEL_FONTSIZE)
#     if show_ylabel:
#         ax.set_ylabel("Memory (GB)", fontsize=YLABEL_FONTSIZE)
#     ax.set_xticks(steps)
#     ax.set_xlim(min(steps), max(steps))
#     ax.margins(x=0)
#     ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
#     if not show_xticklabels:
#         ax.tick_params(axis="x", labelbottom=False)
#     ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
#     ax.grid(False)
#     for s in ("top", "right"):
#         ax.spines[s].set_visible(False)
#     for s in ("left", "bottom"):
#         ax.spines[s].set_color("#444")
#         ax.spines[s].set_linewidth(1.1)


# def build_legend_handles(keys: tuple[tuple[str, str], ...]) -> list[Patch]:
#     """Return legend patches for the requested band keys."""
#     return [
#         Patch(
#             facecolor=PALETTE[k],
#             edgecolor="white",
#             linewidth=0.6,
#             label=_band_label(k),
#         )
#         for k in keys
#     ]


# def _legend_column_major_input_order(
#     row_major_keys: tuple[tuple[str, str], ...],
#     ncols: int,
# ) -> tuple[tuple[str, str], ...]:
#     """Convert desired visual row-major legend order for matplotlib packing."""
#     if ncols <= 0:
#         raise ValueError("legend ncols must be positive")
#     rows = [
#         row_major_keys[start:start + ncols]
#         for start in range(0, len(row_major_keys), ncols)
#     ]
#     ordered = []
#     for col_idx in range(ncols):
#         for row in rows:
#             if col_idx < len(row):
#                 ordered.append(row[col_idx])
#     return tuple(ordered)


# def add_shared_legend(fig, *, layout: str) -> None:
#     legend_items_per_row = LEGEND_ITEMS_PER_ROW_BY_LAYOUT[layout]
#     legend_keys = _legend_column_major_input_order(LEGEND_ORDER, legend_items_per_row)
#     legend_handles = build_legend_handles(legend_keys)
#     legend = fig.legend(
#         handles=legend_handles,
#         labels=[h.get_label() for h in legend_handles],
#         loc="upper left",
#         ncol=legend_items_per_row,
#         frameon=False,
#         bbox_to_anchor=(LEGEND_X_ANCHOR, LEGEND_Y_ANCHOR),
#         fontsize=LEGEND_FONTSIZE,
#         handlelength=LEGEND_HANDLE_LENGTH,
#         handleheight=LEGEND_HANDLE_HEIGHT,
#         columnspacing=LEGEND_COLUMN_SPACING,
#         borderaxespad=0.2,
#     )
#     legend._legend_box.align = "left"


# def _band_label(key: tuple[str, str]) -> str:
#     if key == ("FSDP", "Overheads"):
#         return "FSDP Overheads"
#     return f"{key[0]} — {key[1]}"


# def _stacked_total_max_gb(lines: dict[tuple[str, str], list[float]]) -> float:
#     """Return the maximum (over steps) of the sum of all 8 bands, in GB."""
#     if not lines:
#         return 0.0
#     n = len(next(iter(lines.values())))
#     if n == 0:
#         return 0.0
#     totals = [
#         sum(lines[k][i] for k in STACK_ORDER) / BYTES_PER_GB for i in range(n)
#     ]
#     return float(max(totals))


# def apply_shared_y_ticks(axes, ymin: float, ymax: float) -> None:
#     tick_count = max(2, int(Y_TICK_COUNT))
#     ticks = np.linspace(ymin, ymax, tick_count)
#     for ax in axes.flat:
#         ax.set_ylim(ymin, ymax)
#         ax.set_yticks(ticks)
#         ax.set_yticklabels([
#             Y_TICK_LABEL_FORMAT.format(value=float(tick))
#             for tick in ticks
#         ])


# def make_figure(profile_dir: Path, output: Path, *, layout: str = PANEL_LAYOUT) -> Path:
#     fsdp_dir = find_phase_dir(profile_dir, "baseline")
#     virtu_dir = find_phase_dir(profile_dir, "virtu")
#     fsdp_records = load_phase_records(fsdp_dir)
#     virtu_records = load_phase_records(virtu_dir)

#     fsdp_steps, fsdp_max, fsdp_avg = per_step_lines(fsdp_records)
#     virtu_steps, virtu_max, virtu_avg = per_step_lines(virtu_records)
#     fsdp_r0_steps, fsdp_r0_lines = per_rank_lines(fsdp_records, rank=0)
#     virtu_r0_steps, virtu_r0_lines = per_rank_lines(virtu_records, rank=0)

#     view_data = {
#         "maxrank": ((fsdp_steps,    fsdp_max),       (virtu_steps,    virtu_max)),
#         "avgrank": ((fsdp_steps,    fsdp_avg),       (virtu_steps,    virtu_avg)),
#         "rank0":   ((fsdp_r0_steps, fsdp_r0_lines),  (virtu_r0_steps, virtu_r0_lines)),
#     }
#     unknown = [v for v in VIEWS_TO_RENDER if v not in view_data]
#     if unknown:
#         raise ValueError(
#             f"Unknown VIEWS_TO_RENDER entries: {unknown}. "
#             f"Valid options: {sorted(view_data)}"
#         )
#     if not VIEWS_TO_RENDER:
#         raise ValueError("VIEWS_TO_RENDER is empty — nothing to plot.")

#     plt.rcParams.update({
#         "font.family": FONT_FAMILY,
#         "font.sans-serif": FONT_SANS_SERIF,
#         "mathtext.fontset": MATH_TEXT_FONTSET,
#         "axes.titlesize": TITLE_FONTSIZE,
#         "axes.labelsize": XLABEL_FONTSIZE,
#         "xtick.labelsize": XTICK_FONTSIZE,
#         "ytick.labelsize": YTICK_FONTSIZE,
#         "legend.fontsize": LEGEND_FONTSIZE,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#         "svg.fonttype": "none",
#         "savefig.facecolor": "white",
#         "figure.facecolor": "white",
#     })

#     layout = layout.lower()
#     if layout not in {"stacked", "horizontal"}:
#         raise ValueError("layout must be 'stacked' or 'horizontal'")

#     panel_specs = []
#     for view in VIEWS_TO_RENDER:
#         (fsdp_x, fsdp_y), (virtu_x, virtu_y) = view_data[view]
#         title_fsdp, title_virtu = _VIEW_TITLES[view]
#         panel_specs.append((fsdp_x, fsdp_y, title_fsdp))
#         panel_specs.append((virtu_x, virtu_y, title_virtu))

#     if layout == "horizontal":
#         n_rows = 1
#         n_cols = len(panel_specs)
#         fig, axes = plt.subplots(
#             n_rows,
#             n_cols,
#             figsize=(7.0 * n_cols, 6.3),
#             sharey=True,
#         )
#         axes = np.asarray(axes).reshape(n_rows, n_cols)
#         for col_idx, (steps, lines, title) in enumerate(panel_specs):
#             plot_panel(
#                 axes[0, col_idx],
#                 steps,
#                 lines,
#                 title,
#                 show_xlabel=True,
#                 show_ylabel=(col_idx == 0),
#             )
#     else:
#         n_rows = len(VIEWS_TO_RENDER)
#         n_cols = 2
#         fig, axes = plt.subplots(
#             n_rows,
#             n_cols,
#             figsize=(
#                 STACKED_FIG_WIDTH,
#                 max(STACKED_ROW_HEIGHT * n_rows, STACKED_MIN_HEIGHT),
#             ),
#             sharey=True,
#         )
#         axes = np.asarray(axes).reshape(n_rows, n_cols)
#         for row_idx, view in enumerate(VIEWS_TO_RENDER):
#             (fsdp_x, fsdp_y), (virtu_x, virtu_y) = view_data[view]
#             title_fsdp, title_virtu = _VIEW_TITLES[view]
#             is_last_row = row_idx == n_rows - 1
#             plot_panel(
#                 axes[row_idx, 0],
#                 fsdp_x,
#                 fsdp_y,
#                 title_fsdp,
#                 show_xlabel=is_last_row,
#                 show_ylabel=True,
#                 show_xticklabels=is_last_row,
#             )
#             plot_panel(
#                 axes[row_idx, 1],
#                 virtu_x,
#                 virtu_y,
#                 title_virtu,
#                 show_xlabel=is_last_row,
#                 show_ylabel=False,
#                 show_xticklabels=is_last_row,
#             )

#     # Single shared y-axis range across every rendered panel. The biggest
#     # stack sets the headroom; smaller stacks then visually show savings
#     # against the same scale.
#     rendered_lines = []
#     for view in VIEWS_TO_RENDER:
#         (_, fsdp_y), (_, virtu_y) = view_data[view]
#         rendered_lines.append(fsdp_y)
#         rendered_lines.append(virtu_y)
#     global_ymax_gb = max(
#         (_stacked_total_max_gb(lines) for lines in rendered_lines),
#         default=0.0,
#     )
#     ymax_padded = global_ymax_gb * 1.08 if global_ymax_gb > 0 else 1.0
#     apply_shared_y_ticks(axes, 0.0, ymax_padded)

#     add_shared_legend(fig, layout=layout)
#     legend_top_reserve = LEGEND_TOP_RESERVE_BY_LAYOUT[layout]

#     # Reserve enough figure height for the active legend layout so the panels
#     # don't grow into it.
#     fig.tight_layout(rect=[0, 0.0, 1, 1.0 - legend_top_reserve])

#     output.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output, bbox_inches="tight", dpi=200)
#     png_out = output.with_suffix(".png")
#     fig.savefig(png_out, bbox_inches="tight", dpi=200)
#     plt.close(fig)
#     return output


# def find_phase_dir(profile_dir: Path, phase_name: str) -> Path:
#     candidates = []
#     if phase_name == "baseline":
#         candidates.extend([profile_dir / "fsdp", profile_dir / "baseline"])
#     else:
#         candidates.append(profile_dir / phase_name)
#     for candidate in candidates:
#         if candidate.is_dir():
#             return candidate
#     return candidates[0]


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument(
#         "--profile-dir",
#         type=Path,
#         required=True,
#         help="Directory containing fsdp/ and virtu/ subdirs with "
#              "memory_breakdown_rank*.jsonl files.",
#     )
#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=None,
#         help="Output PDF path (default: <profile-dir>/component_memory_breakdown.pdf).",
#     )
#     parser.add_argument(
#         "--layout",
#         choices=("stacked", "horizontal"),
#         default=PANEL_LAYOUT,
#         help=(
#             "Panel layout: stacked = one row per view with FSDP/Virtu columns; "
#             "horizontal = one row with all panels left-to-right."
#         ),
#     )
#     args = parser.parse_args()

#     profile_dir: Path = args.profile_dir.resolve()
#     output: Path = (
#         args.output.resolve()
#         if args.output is not None
#         else profile_dir / "component_memory_breakdown.pdf"
#     )
#     saved = make_figure(profile_dir, output, layout=args.layout)
#     print(f"Saved: {saved}")
#     print(f"Saved: {saved.with_suffix('.png')}")


# if __name__ == "__main__":
#     main()



"""Plot per-step component memory breakdown for FSDP (baseline) vs Virtu.

Reads `memory_breakdown_rank{0..N}.jsonl` from `<profile-dir>/fsdp/` (or
`<profile-dir>/baseline/`) and `<profile-dir>/virtu/`, snapshots each step at
phase=`before_optimizer_step`, then renders each panel as a *stacked band
chart* with 8 component bands per panel (in a fixed order shared across
panels):

    PLE     | Weight, Gradient, Optimizer State, Activation
    Others  | Weight, Gradient, Optimizer State, Activation

PLE = `ple_table` + `ple_dense_side`.
Others = every other component (mlp, attention_full/sliding, norms, other_params,
token_embedding_lm_head_shared, vision_projector, audio).

Activation accounting:
- `PLE — Activation` = `ple_activation_bytes` field, populated by forward
  pre/post hooks installed by `MemoryBreakdownProfiler` on PLE submodules
  (sum of `memory_allocated()` deltas around their forward calls during the
  step's first forward pass).
- `Others — Activation` = (`peak_allocated_since_step_begin` − persistent
  weight+grad+opt across all components) − `PLE — Activation`. This is a
  residual that captures the rest of the within-step activation watermark.
- The 8 lines therefore sum to the peak (within-step) allocated bytes — the
  OOM-relevant memory pressure, not the steady-state snapshot.

Renders 4 panels (2x2):
    [FSDP Max Peak]   [FSDP Avg Peak]
    [Virtu Max Peak]  [Virtu Avg Peak]

where Max/Avg are taken across ranks at the chosen phase using
`peak_allocated_since_step_begin`.

Color palette is sampled from `agent/overleaf/[NeurIPS'26] Efficient PLE/figs/method.pdf`:
warm hues (peach -> red) for PLE, cool hues (sage -> navy) for Others.

Usage:
python scripts/plots/plot_memory_breakdown.py --profile-dir output/memory_profile/model-gemma-4-E4B-it_data-dolma3_50k.json_steps-10_lr-1e-5_bs-4_pad-max-batch_msl-4096_gpus-4_ga-1_wd-0.0_seed-42
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHASE = "before_optimizer_step"
BYTES_PER_GB = 1024 ** 3  # GiB; labelled as "GB" to match common ML usage

PLE_COMPONENTS = {"ple_table", "ple_dense_side"}
GROUPS = ("PLE", "Others")
CATEGORIES = ("Weight", "Gradient", "Optimizer", "Activation")
KIND_KEY = {"Weight": "weight", "Gradient": "grad", "Optimizer": "optimizer_state"}

# =============================================================================
# USER-TUNABLE: per-band fill colours (hex). Edit any row to recolour that band
# in every panel; the legend swatch follows automatically. Order is irrelevant
# here — the bottom-to-top stacking order is set separately by STACK_ORDER.
# Defaults sampled from agent/samples/Screenshot 2026-05-06 at 8.02.55 PM.png.
# =============================================================================
# PALETTE: dict[tuple[str, str], str] = {
#     ("PLE",    "Weight"):     "#93C18E",  # sage green   (screenshot #8, bottom band)
#     ("PLE",    "Gradient"):   "#F1D9AE",  # cream peach  (screenshot #7)
#     ("PLE",    "Optimizer"):  "#A93232",  # deep red     (screenshot #6)
#     ("PLE",    "Activation"): "#EFA9A8",  # salmon pink  (screenshot #5)
#     ("Others", "Weight"):     "#6FACD7",  # sky blue     (screenshot #4)
#     ("Others", "Gradient"):   "#6E3D2A",  # dark brown   (screenshot #3)
#     ("Others", "Optimizer"):  "#DDDDDD",  # light gray   (screenshot #2)
#     ("Others", "Activation"): "#B89BCF",  # lavender
#     # FSDP-overheads band: peak − every directly measured component above.
#     # Captures FSDP all-gather scratch (transient unsharded params), optimizer
#     # step temporaries, and allocator caching. This is the slack Virtu
#     # eliminates by keeping the PLE table out of FSDP's flat-buffer.
#     ("FSDP",   "Overheads"):  "#F2A8B0",  # salmon (screenshot #1, top band)
# }
PALETTE: dict[tuple[str, str], str] = {
    ("PLE",    "Weight"):     "#F3C48E",  # method-draft green row tiles
    ("PLE",    "Gradient"):   "#337296",  # method-draft orange key/gradient boxes
    ("PLE",    "Optimizer"):  "#9D281E",  # method-draft purple optimizer states
    ("PLE",    "Activation"): "#7C8354",  # method-draft yellow tau/state boxes
    ("Others", "Weight"):     "#BFDBE8",  # method-draft pale blue lookup/transfer tint
    ("Others", "Gradient"):   "#F1F1F1",  # method-draft blue communication arrows
    
    ("Others", "Optimizer"):  "#FFD59A",  # method-draft inactive/neutral rows
    ("Others", "Activation"): "#D7EEF4",  # method-draft green checks/active flow
    # FSDP-overheads band: peak − every directly measured component above.
    # Captures FSDP all-gather scratch (transient unsharded params), optimizer
    # step temporaries, and allocator caching. This is the slack Virtu
    # eliminates by keeping the PLE table out of FSDP's flat-buffer.
    ("FSDP",   "Overheads"):  "#949F75",  # method-draft red hot/congestion highlight
}

# =============================================================================
# USER-TUNABLE: font sizes (in points). Bump any value to enlarge that element
# across all panels; matplotlib resolves these at draw time, so re-running the
# script with edited numbers is enough.
# =============================================================================
def configured_sans_serif_stack() -> list[str]:
    """Register optional Helvetica Neue font files and return fallback stack."""
    font_paths: list[Path] = []
    font_path_env = os.environ.get("HELVETICA_NEUE_FONT_PATH")
    if font_path_env:
        font_paths.extend(Path(item).expanduser() for item in font_path_env.split(os.pathsep) if item)

    font_dir_env = os.environ.get("HELVETICA_NEUE_FONT_DIR")
    if font_dir_env:
        font_paths.extend(
            Path(path)
            for path in font_manager.findSystemFonts(
                fontpaths=[str(Path(font_dir_env).expanduser())],
                fontext="ttf",
            )
        )

    families: list[str] = []
    for font_path in font_paths:
        if not font_path.exists():
            raise SystemExit(f"Configured Helvetica Neue font does not exist: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        families.append(font_manager.FontProperties(fname=str(font_path)).get_name())

    fallback = [
        "Helvetica Neue",
        "Neue Helvetica",
        "Helvetica Neue LT Pro",
        "Helvetica",
        "Nimbus Sans",
        "Arial",
        "DejaVu Sans",
    ]
    return list(dict.fromkeys(families + fallback))


FONT_FAMILY          = "sans-serif"
FONT_SANS_SERIF      = configured_sans_serif_stack()
MATH_TEXT_FONTSET    = "dejavusans"
TITLE_FONTSIZE       = 30    # per-panel title (e.g. "FSDP — Max Peak")
TITLE_PAD            = 14    # vertical gap between each panel title and axes
XLABEL_FONTSIZE      = 30    # x-axis label ("Step")
YLABEL_FONTSIZE      = 30    # y-axis label ("Memory (GB)")
XTICK_FONTSIZE       = 28    # x-axis tick numerals
YTICK_FONTSIZE       = 28    # y-axis tick numerals
X_TICK_STEP          = 200   # show x-axis tick numerals every N training steps
VIRTU_SMOOTH_WINDOW  = 25    # centered moving average window for Virtu panels
Y_TICK_COUNT         = 3     # linspace ticks from y-min to y-max; 3 -> 0, mid, top
Y_TICK_LABEL_FORMAT  = "{value:.0f}"
LEGEND_FONTSIZE      = 26    # legend text
LEGEND_HANDLE_LENGTH = 1.5   # legend swatch width  (matplotlib units)
LEGEND_HANDLE_HEIGHT = 1.2   # legend swatch height (matplotlib units)
LEGEND_COLUMN_SPACING = 1.4  # horizontal gap between legend entries
LEGEND_TOP_RESERVE   = 0.30 # fraction of figure height reserved above panels
                             # for the legend. Bump this up if the legend
                             # collides with the first row's title; trim it
                             # down if there is too much empty space above.
LEGEND_X_ANCHOR      = 0.12 # left edge of the shared legend in figure coords
LEGEND_Y_ANCHOR      = 0.99  # top edge of the shared legend in figure coords
LEGEND_ITEMS_PER_ROW = 5     # 5 columns -> 2 rows for the 9 memory bands

# =============================================================================
# USER-TUNABLE: which per-step views to render. Each entry becomes a row in
# the figure (FSDP in the left column, Virtu in the right). Valid options:
#   "maxrank" — breakdown of the rank with the largest peak that step
#   "avgrank" — per-component mean across ranks
#   "rank0"   — rank 0's own breakdown (single rank, no aggregation)
# Order matters: rows are rendered top-to-bottom in the order listed.
# Examples:
#   VIEWS_TO_RENDER = ("maxrank", "avgrank")              # 2 rows, 4 panels
#   VIEWS_TO_RENDER = ("rank0",)                          # 1 row, 2 panels
#   VIEWS_TO_RENDER = ("maxrank", "avgrank", "rank0")     # 3 rows, 6 panels
# =============================================================================
# VIEWS_TO_RENDER: tuple[str, ...] = ("rank0",) # ("maxrank", "avgrank", "rank0")
VIEWS_TO_RENDER: tuple[str, ...] = ("rank0", "avgrank") # ("maxrank", "avgrank", "rank0")

# Figure panel layout:
#   "stacked"    -> one row per view, two columns: FSDP | Virtu
#   "horizontal" -> one row with all panels left-to-right: FSDP, Virtu, ...
PANEL_LAYOUT = "horizontal"

_VIEW_TITLES = {
    "maxrank": ("FSDP — Max Peak (across ranks)", "Virtu — Max Peak (across ranks)"),
    "avgrank": ("FSDP — Avg across Ranks", "Virtu — Avg across Ranks"),
    "rank0":   ("FSDP — Rank 0",                  "Virtu — Rank 0"),
}

# Panel-level accent: matches method.pdf's "problem (red) vs solution (green)" framing.
BACKEND_ACCENT = {"FSDP": "#8C2A2A", "Virtu": "#3F7F5A"}

# Canonical bottom-to-top stack order. The same order is used in every panel
# (PLE bands at the bottom, Others on top; W -> G -> Opt -> Act within each
# group), so visually identical y-positions across panels mean identical
# components.
STACK_ORDER: tuple[tuple[str, str], ...] = (
    ("PLE", "Weight"),
    ("PLE", "Gradient"),
    ("PLE", "Optimizer"),
    ("PLE", "Activation"),
    ("Others", "Weight"),
    ("Others", "Gradient"),
    ("Others", "Optimizer"),
    ("Others", "Activation"),
    ("FSDP", "Overheads"),  # 9th band — FSDP scratch + opt temps + allocator misc
)

# Top legend order, independent of the bottom-to-top stack order. This is the
# desired visual row-major order; _legend_column_major_input_order converts it
# for matplotlib's column-major legend packing.
LEGEND_ORDER: tuple[tuple[str, str], ...] = (
    ("Others", "Weight"),
    ("Others", "Gradient"),
    ("Others", "Optimizer"),
    ("Others", "Activation"),
    ("FSDP", "Overheads"),
    ("PLE", "Weight"),
    ("PLE", "Gradient"),
    ("PLE", "Optimizer"),
    ("PLE", "Activation"),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase_records(profile_dir: Path, phase: str = PHASE) -> list[dict]:
    """Return detailed rank-jsonl records matching `phase`, across all ranks."""
    records: list[dict] = []
    files = sorted(profile_dir.glob("memory_breakdown_rank*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No memory_breakdown_rank*.jsonl files under {profile_dir}"
        )
    for jsonl in files:
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["phase"] == phase and rec["detailed"] is True:
                    records.append(rec)
    if not records:
        raise ValueError(f"No detailed records with phase={phase!r} in {profile_dir}")
    return records


def breakdown_for_record(rec: dict) -> dict[tuple[str, str], int]:
    """Return {(group, category): bytes} for one (rank, step) record.

    Now produces 9 bands:
    - 6 directly measured persistent bands (PLE/Others × Weight/Grad/Opt) from
      `component_bytes`.
    - PLE-Activation from the profiler's PLE forward-hook accumulator
      (`ple_activation_bytes`).
    - Others-Activation from the profiler's non-PLE forward-hook accumulator
      (`others_activation_bytes`).
    - Misc-Residual = `peak_allocated_since_step_begin` minus all of the above
      — captures FSDP all-gather scratch + optimizer-step temps + allocator
      misc that aren't directly attributable.

    Bands sum to the within-step peak.
    """
    out: dict[tuple[str, str], int] = {(g, c): 0 for g in GROUPS for c in CATEGORIES}
    out[("FSDP", "Overheads")] = 0

    persistent = 0
    comps = rec["component_bytes"]
    for comp_name, kinds in comps.items():
        group = "PLE" if comp_name in PLE_COMPONENTS else "Others"
        for cat in ("Weight", "Gradient", "Optimizer"):
            v = int(kinds[KIND_KEY[cat]])
            out[(group, cat)] += v
            persistent += v

    ple_activation = max(0, int(rec["ple_activation_bytes"] or 0))
    others_activation = max(0, int(rec["others_activation_bytes"] or 0))
    peak = int(rec["peak_allocated_since_step_begin"])
    residual = max(0, peak - persistent - ple_activation - others_activation)

    out[("PLE", "Activation")] = ple_activation
    out[("Others", "Activation")] = others_activation
    out[("FSDP", "Overheads")] = residual
    return out


def per_rank_lines(records: list[dict], rank: int = 0) -> tuple[list[int], dict]:
    """Per-step breakdown for a single rank only.

    Returns (steps, lines) where lines maps (group, category) -> list of
    bytes aligned with `steps`, taken from the chosen rank's records.
    """
    by_step: dict[int, dict] = {}
    for rec in records:
        if int(rec["rank"]) != rank:
            continue
        by_step[int(rec["step"])] = breakdown_for_record(rec)

    steps = sorted(by_step)
    keys = list(STACK_ORDER)
    lines = {k: [float(by_step[s][k]) for s in steps] for k in keys}
    return steps, lines


def per_step_lines(records: list[dict]) -> tuple[list[int], dict, dict]:
    """Aggregate records into max-rank and avg-across-ranks per step.

    Returns (steps, max_lines, avg_lines) where each *_lines dict maps
    (group, category) -> list[float bytes] aligned with `steps`.

    Definitions (matching `memory_step_summary.md` semantics on the totals):
    - Max: pick the rank with the largest `peak_allocated_since_step_begin` for that step,
      then take its per-component breakdown. Lines sum to the max-rank
      peak, so the decomposition is physically realisable on a single GPU.
      (Naively taking per-component max-across-ranks would over-count, since
      different components can peak on different ranks.)
    - Avg: per-component mean across ranks. Lines sum to mean peak.
    """
    by_step_rank: dict[int, dict[int, tuple[dict, int]]] = defaultdict(dict)
    for rec in records:
        step = int(rec["step"])
        rank = int(rec["rank"])
        by_step_rank[step][rank] = (
            breakdown_for_record(rec),
            int(rec["peak_allocated_since_step_begin"]),
        )

    steps = sorted(by_step_rank)
    keys = list(STACK_ORDER)
    max_lines = {k: [] for k in keys}
    avg_lines = {k: [] for k in keys}
    for step in steps:
        per_rank = list(by_step_rank[step].values())
        # Avg: per-component mean across ranks.
        for k in keys:
            avg_lines[k].append(float(np.mean([b[k] for b, _ in per_rank])))
        # Max: breakdown of the rank with the highest peak allocation.
        max_breakdown, _ = max(per_rank, key=lambda br_alloc: br_alloc[1])
        for k in keys:
            max_lines[k].append(float(max_breakdown[k]))
    return steps, max_lines, avg_lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(ax, steps, lines, title, *, show_xlabel=True, show_ylabel=True):
    """Stacked area panel: 8 colored bands stacked bottom-to-top in STACK_ORDER.

    Title is rendered as a plain panel title above the axes (no banner / no
    italic / no bold) to match the requested aesthetic. Y-axis is memory in
    GB; band heights at each step sum to the within-step peak.
    """
    band_data = np.asarray(
        [np.asarray(lines[k]) / BYTES_PER_GB for k in STACK_ORDER]
    )
    colors = [PALETTE[k] for k in STACK_ORDER]
    labels = [_band_label(k) for k in STACK_ORDER]

    ax.stackplot(
        steps,
        band_data,
        colors=colors,
        labels=labels,
        edgecolor="white",
        linewidth=0.4,
        antialiased=True,
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=XLABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Memory (GB)", fontsize=YLABEL_FONTSIZE)
    min_step = int(min(steps))
    max_step = int(max(steps))
    tick_start = 0 if min_step <= X_TICK_STEP else int(np.floor(min_step / X_TICK_STEP) * X_TICK_STEP)
    tick_end = int(np.ceil(max_step / X_TICK_STEP) * X_TICK_STEP)
    ax.set_xticks(np.arange(tick_start, tick_end + 1, X_TICK_STEP))
    ax.set_xlim(tick_start, max_step)
    ax.margins(x=0)
    ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
        ax.spines[s].set_linewidth(1.1)


def build_legend_handles(keys: tuple[tuple[str, str], ...]) -> list[Patch]:
    """Return legend patches for the requested band keys."""
    return [
        Patch(
            facecolor=PALETTE[k],
            edgecolor="white",
            linewidth=0.6,
            label=_band_label(k),
        )
        for k in keys
    ]


def _legend_column_major_input_order(
    row_major_keys: tuple[tuple[str, str], ...],
    ncols: int,
) -> tuple[tuple[str, str], ...]:
    """Convert desired visual row-major legend order for matplotlib packing."""
    if ncols <= 0:
        raise ValueError("legend ncols must be positive")
    rows = [
        row_major_keys[start:start + ncols]
        for start in range(0, len(row_major_keys), ncols)
    ]
    ordered = []
    for col_idx in range(ncols):
        for row in rows:
            if col_idx < len(row):
                ordered.append(row[col_idx])
    return tuple(ordered)


def add_shared_legend(fig) -> None:
    legend_keys = _legend_column_major_input_order(LEGEND_ORDER, LEGEND_ITEMS_PER_ROW)
    legend_handles = build_legend_handles(legend_keys)
    legend = fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="upper left",
        ncol=LEGEND_ITEMS_PER_ROW,
        frameon=False,
        bbox_to_anchor=(LEGEND_X_ANCHOR, LEGEND_Y_ANCHOR),
        fontsize=LEGEND_FONTSIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        columnspacing=LEGEND_COLUMN_SPACING,
        borderaxespad=0.2,
    )
    legend._legend_box.align = "left"


def _band_label(key: tuple[str, str]) -> str:
    if key == ("FSDP", "Overheads"):
        return "FSDP Overheads"
    return f"{key[0]} — {key[1]}"


def _stacked_total_max_gb(lines: dict[tuple[str, str], list[float]]) -> float:
    """Return the maximum (over steps) of the sum of all 8 bands, in GB."""
    if not lines:
        return 0.0
    n = len(next(iter(lines.values())))
    if n == 0:
        return 0.0
    totals = [
        sum(lines[k][i] for k in STACK_ORDER) / BYTES_PER_GB for i in range(n)
    ]
    return float(max(totals))


def _smooth_values(values: list[float], window: int) -> list[float]:
    window = int(window)
    if window <= 1 or len(values) < 3:
        return list(values)
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return list(values)
    arr = np.asarray(values, dtype=float)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid").tolist()


def smooth_lines(
    lines: dict[tuple[str, str], list[float]],
    window: int,
) -> dict[tuple[str, str], list[float]]:
    return {key: _smooth_values(values, window) for key, values in lines.items()}


def apply_shared_y_ticks(axes, ymin: float, ymax: float) -> None:
    tick_count = max(2, int(Y_TICK_COUNT))
    ticks = np.linspace(ymin, ymax, tick_count)
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(ticks)
        ax.set_yticklabels([
            Y_TICK_LABEL_FORMAT.format(value=float(tick))
            for tick in ticks
        ])


def make_figure(profile_dir: Path, output: Path, *, layout: str = PANEL_LAYOUT) -> Path:
    fsdp_dir = find_phase_dir(profile_dir, "baseline")
    virtu_dir = find_phase_dir(profile_dir, "virtu")
    fsdp_records = load_phase_records(fsdp_dir)
    virtu_records = load_phase_records(virtu_dir)

    fsdp_steps, fsdp_max, fsdp_avg = per_step_lines(fsdp_records)
    virtu_steps, virtu_max, virtu_avg = per_step_lines(virtu_records)
    fsdp_r0_steps, fsdp_r0_lines = per_rank_lines(fsdp_records, rank=0)
    virtu_r0_steps, virtu_r0_lines = per_rank_lines(virtu_records, rank=0)

    if VIRTU_SMOOTH_WINDOW > 1:
        virtu_max = smooth_lines(virtu_max, VIRTU_SMOOTH_WINDOW)
        virtu_avg = smooth_lines(virtu_avg, VIRTU_SMOOTH_WINDOW)
        virtu_r0_lines = smooth_lines(virtu_r0_lines, VIRTU_SMOOTH_WINDOW)

    view_data = {
        "maxrank": ((fsdp_steps,    fsdp_max),       (virtu_steps,    virtu_max)),
        "avgrank": ((fsdp_steps,    fsdp_avg),       (virtu_steps,    virtu_avg)),
        "rank0":   ((fsdp_r0_steps, fsdp_r0_lines),  (virtu_r0_steps, virtu_r0_lines)),
    }
    unknown = [v for v in VIEWS_TO_RENDER if v not in view_data]
    if unknown:
        raise ValueError(
            f"Unknown VIEWS_TO_RENDER entries: {unknown}. "
            f"Valid options: {sorted(view_data)}"
        )
    if not VIEWS_TO_RENDER:
        raise ValueError("VIEWS_TO_RENDER is empty — nothing to plot.")

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": FONT_SANS_SERIF,
        "mathtext.fontset": MATH_TEXT_FONTSET,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": XLABEL_FONTSIZE,
        "xtick.labelsize": XTICK_FONTSIZE,
        "ytick.labelsize": YTICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    layout = layout.lower()
    if layout not in {"stacked", "horizontal"}:
        raise ValueError("layout must be 'stacked' or 'horizontal'")

    panel_specs = []
    for view in VIEWS_TO_RENDER:
        (fsdp_x, fsdp_y), (virtu_x, virtu_y) = view_data[view]
        title_fsdp, title_virtu = _VIEW_TITLES[view]
        panel_specs.append((fsdp_x, fsdp_y, title_fsdp))
        panel_specs.append((virtu_x, virtu_y, title_virtu))

    if layout == "horizontal":
        n_rows = 1
        n_cols = len(panel_specs)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7.0 * n_cols, 6.3),
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        for col_idx, (steps, lines, title) in enumerate(panel_specs):
            plot_panel(
                axes[0, col_idx],
                steps,
                lines,
                title,
                show_xlabel=True,
                show_ylabel=(col_idx == 0),
            )
    else:
        n_rows = len(VIEWS_TO_RENDER)
        n_cols = 2
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, max(4.5 * n_rows, 5.0)),
            sharey=True,
        )
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        for row_idx, view in enumerate(VIEWS_TO_RENDER):
            (fsdp_x, fsdp_y), (virtu_x, virtu_y) = view_data[view]
            title_fsdp, title_virtu = _VIEW_TITLES[view]
            is_last_row = row_idx == n_rows - 1
            plot_panel(
                axes[row_idx, 0],
                fsdp_x,
                fsdp_y,
                title_fsdp,
                show_xlabel=is_last_row,
                show_ylabel=True,
            )
            plot_panel(
                axes[row_idx, 1],
                virtu_x,
                virtu_y,
                title_virtu,
                show_xlabel=is_last_row,
                show_ylabel=False,
            )

    # Single shared y-axis range across every rendered panel. The biggest
    # stack sets the headroom; smaller stacks then visually show savings
    # against the same scale.
    rendered_lines = []
    for view in VIEWS_TO_RENDER:
        (_, fsdp_y), (_, virtu_y) = view_data[view]
        rendered_lines.append(fsdp_y)
        rendered_lines.append(virtu_y)
    global_ymax_gb = max(
        (_stacked_total_max_gb(lines) for lines in rendered_lines),
        default=0.0,
    )
    ymax_padded = global_ymax_gb * 1.08 if global_ymax_gb > 0 else 1.0
    apply_shared_y_ticks(axes, 0.0, ymax_padded)

    add_shared_legend(fig)
    legend_top_reserve = LEGEND_TOP_RESERVE

    # Reserve enough figure height for the active legend layout so the panels
    # don't grow into it.
    fig.tight_layout(rect=[0, 0.0, 1, 1.0 - legend_top_reserve])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    png_out = output.with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output


def find_phase_dir(profile_dir: Path, phase_name: str) -> Path:
    candidates = []
    if phase_name == "baseline":
        candidates.extend([profile_dir / "fsdp", profile_dir / "baseline"])
    else:
        candidates.append(profile_dir / phase_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        required=True,
        help="Directory containing fsdp/ and virtu/ subdirs with "
             "memory_breakdown_rank*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <profile-dir>/component_memory_breakdown.pdf).",
    )
    parser.add_argument(
        "--layout",
        choices=("stacked", "horizontal"),
        default=PANEL_LAYOUT,
        help=(
            "Panel layout: stacked = one row per view with FSDP/Virtu columns; "
            "horizontal = one row with all panels left-to-right."
        ),
    )
    args = parser.parse_args()

    profile_dir: Path = args.profile_dir.resolve()
    output: Path = (
        args.output.resolve()
        if args.output is not None
        else profile_dir / "component_memory_breakdown.pdf"
    )
    saved = make_figure(profile_dir, output, layout=args.layout)
    print(f"Saved: {saved}")
    print(f"Saved: {saved.with_suffix('.png')}")


if __name__ == "__main__":
    main()

