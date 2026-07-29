#!/usr/bin/env python3
"""Figure-2 row (§2.4): m2a / m2b / m3 / m4 — UNIFORM PHYSICAL SIZE.

Kevin 2026-07-26: the four sub-figures must render at the SAME height.
Invariant here: every figure is EXACTLY FIGSIZE = (9.9, 5.5) inches with the
same font constants — include each at equal column width and the rendered
heights are identical by construction. Data = the MEASURED blocks in
motivation_v2_plots.md (2026-07-26); no re-benching here.

Also per feedback: m2a renamed (no "case study"); m2b gains the memory panel.
Old m2a filename is still emitted (same content) so an un-updated \\includegraphics
doesn't break; delete once the tex switches.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from constants import (BAR_NAVY, BAR_NEUTRAL, BAR_NEUTRAL_CYAN, BAR_TEAL,
                       FONT_SIZE_BASE, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
                       FONT_SIZE_AXIS_TITLE, GRID_COLOR, SPINE_COLOR,
                       HELVETICA_LIKE_FONTS)

# ═════════════ TUNABLES (box geometry — hand-tune here) ═════════════════════
# Canvas (inches). Same for all five figures; LaTeX includes at equal width,
# so rendered height scales with FIG_H/FIG_W.
FIG_W = 9.9
FIG_H = 3.09
FIGSIZE = (FIG_W, FIG_H)
# The AXES BOX (frame) — fractions of the canvas, same for every figure:
#   box HEIGHT (in) = (BOX_TOP - BOX_BOTTOM) * FIG_H      [now: 0.59*3.09 = 1.82in]
#   raise BOX_TOP    -> taller box (eats the legend band above)
#   lower BOX_BOTTOM -> taller box (eats the x-tick strip below)
# Box WIDTH per panel (in) = (BOX_RIGHT - BOX_LEFT)/(n + (n-1)*WSPACE) * FIG_W
#   n = panels in that figure (2 everywhere, 3 for m3); WSPACE is the gap
#   between panels in fractions of one panel's width.
BOX_LEFT   = 0.13    # left margin (y-axis titles live here)
BOX_RIGHT  = 0.985
BOX_BOTTOM = 0.17    # x-tick strip below the box
BOX_TOP    = 0.76    # legend band above the box
WSPACE_2P  = 0.58    # panel gap, 2-panel figures (m2a/m2b/m4/m6)
WSPACE_3P  = 0.75    # panel gap, m3 (3 panels)
BOX_LEFT_3P = 0.105  # m3 uses a slimmer left margin
# Bars: one width for figs 2/3/6 in the shared 2-group frame.
# ════════════════════════════════════════════════════════════════════════════
OUT = __import__("pathlib").Path(__file__).resolve().parent / "out"

# Kevin 2026-07-26: figures 2-6 run 2pt below house sizes (house: ticks 22,
# legend 22, axis titles 26) — at the 4.125in standard the house sizes crowd.
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": HELVETICA_LIKE_FONTS,
    "font.size": FONT_SIZE_BASE, "xtick.labelsize": FONT_SIZE_TICK - 2,
    "ytick.labelsize": FONT_SIZE_TICK - 2, "legend.fontsize": FONT_SIZE_LEGEND - 2,
    "axes.labelsize": FONT_SIZE_AXIS_TITLE - 2, "axes.edgecolor": SPINE_COLOR,
    "axes.grid": True, "grid.color": GRID_COLOR, "axes.grid.axis": "y",
    "axes.axisbelow": True, "pdf.fonttype": 42,
})


# Uniform AXES BOX across all five figures (Kevin 2026-07-26): same bottom/top
# margin FRACTIONS everywhere -> the frame rectangle is byte-identical in
# height (box = (TOP-BOTTOM) x 3.09in = 1.82in). Legends live in the reserved
# band above the frames. Only left/wspace vary with panel count/label width.
BOTTOM, TOP = BOX_BOTTOM, BOX_TOP


def make_fig(n, left=BOX_LEFT, right=BOX_RIGHT, wspace=WSPACE_2P):
    fig, axes = plt.subplots(1, n, figsize=FIGSIZE)
    fig.subplots_adjust(left=left, right=right, bottom=BOTTOM, top=TOP,
                        wspace=wspace)
    return fig, (axes if n > 1 else [axes])


def band_legend(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.005), borderaxespad=0.0,
               handlelength=1.3, columnspacing=1.2)


def _style(ax):
    ax.grid(axis="x", visible=False)
    for s in ax.spines.values():
        s.set_color(SPINE_COLOR)


BAR_W = 0.24            # THE bar width (data units) — figs 2/3 match fig 6
GROUP_XLIM = (-0.6, 1.6)  # 2-group frame shared by m2a/m2b/m6


def _bars(ax, groups, series, width=BAR_W, log=False, xlim=None):
    x = np.arange(len(groups))
    n = len(series)
    for i, (label, vals, color) in enumerate(series):
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, vals, width, label=label, color=color,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(x, groups)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif len(groups) == 2:
        ax.set_xlim(*GROUP_XLIM)
    if log:
        ax.set_yscale("log")
    _style(ax)


def save(fig, *names):
    fig.set_size_inches(*FIGSIZE)      # belt: enforce the standard
    for ax in fig.axes:
        b = ax.get_position()
        print(f"  [box] {names[0]}: y0={b.y0:.3f} h={b.height:.3f} "
              f"(={b.height * FIGSIZE[1]:.3f}in)")
    for n in names:
        fig.savefig(OUT / f"{n}.pdf")
        fig.savefig(OUT / f"{n}.png", dpi=150)
    plt.close(fig)


# ── m2a — adapter-GEMM placement microbenchmark (2 panels) ─────────────────
fig, (aL, aR) = make_fig(2)
S = [("Resident", [0.053, 0.402], BAR_NEUTRAL),
     ("Staged",   [1.651, 13.080], BAR_NEUTRAL_CYAN),
     ("Streamed (ours)", [1.598, 12.701], BAR_NAVY)]
_bars(aL, ["32K rows", "256K rows"], S, log=True)
aL.set_ylabel("time (ms)")
M = [("Resident", [0.336, 2.684], BAR_NEUTRAL),
     ("Staged",   [0.336, 2.684], BAR_NEUTRAL_CYAN),
     ("Streamed (ours)", [0.0007, 0.0007], BAR_NAVY)]
_bars(aR, ["32K rows", "256K rows"], M)
aR.set_ylabel("input bytes (GB)")
for gx in (0, 1):
    aR.text(gx + BAR_W, 0.06, "≈0", ha="center", fontsize=FONT_SIZE_BASE - 2,
            color=BAR_NAVY, fontweight="bold")
h, l = aL.get_legend_handles_labels()
band_legend(fig, h, l, 3)
save(fig, "m2a-adapter-gemm-placement", "m2a-kernel-case-study")

# ── m2b — scatter fusion: time AND memory (2 panels) ────────────────────────
fig, (bL, bR) = make_fig(2)
S = [("Unfused", [41.93, 43.98], BAR_NEUTRAL),
     ("Fused (ours)", [36.38, 49.06], BAR_NAVY)]
_bars(bL, ["Fwd", "Gate/up grad"], S)
bL.set_ylabel("time (ms)")
GATHER, OUTCPY = 4.29, 8.59
M = [("Unfused", [GATHER + OUTCPY, GATHER + OUTCPY], BAR_NEUTRAL),
     ("Fused (ours)", [GATHER, GATHER], BAR_NAVY)]
_bars(bR, ["Fwd", "Gate/up grad"], M)
bR.set_ylabel("copies (GB)")
h, l = bL.get_legend_handles_labels()
band_legend(fig, h, l, 2)
save(fig, "m2b-scatter-fused")

# ── m3 — per-module composition (3 panels) ──────────────────────────────────
fig, (c1, c2, c3) = make_fig(3, left=BOX_LEFT_3P, wspace=WSPACE_3P)
POL = [("Recompute", BAR_NEUTRAL), ("Offload", BAR_NEUTRAL_CYAN),
       ("Composed (ours)", BAR_NAVY)]
grp = ["Attn", "MLP"]
peak = [[30.5, 44.0], [30.5, 26.6], [36.5, 31.5]]
link = [[0.0, 0.0], [30.5, 76.3], [14.6, 51.3]]
fwd  = [[333, 251], [310, 251], [305, 250]]
bwd  = [[1410, 693], [1634, 1360], [1578, 1230]]
_bars(c1, grp, [(n, v, c) for (n, c), v in zip(POL, peak)])
c1.set_ylabel("peak (GiB)")
_bars(c2, grp, [(n, v, c) for (n, c), v in zip(POL, link)])
c2.set_ylabel("link (GiB)")
x = np.arange(len(grp))
for i, ((name, color), f, b) in enumerate(zip(POL, fwd, bwd)):
    off = (i - 1) * 0.24
    c3.bar(x + off, f, 0.24, color=color, alpha=0.45, edgecolor="white",
           linewidth=0.8, zorder=3)
    c3.bar(x + off, b, 0.24, bottom=f, color=color, edgecolor="white",
           linewidth=0.8, zorder=3)
c3.set_xticks(x, grp)
c3.set_xlim(*GROUP_XLIM)
_style(c3)
c3.set_ylabel("time (ms)")
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in POL]
band_legend(fig, h, [n for n, _ in POL], 3)
save(fig, "m3-composition")

# ── m4 — Grace-kernel placement microbenchmark (2 panels) ───────────────────
fig, (dL, dR) = make_fig(2)
S = [("CPU (ours)", [1.36, 1.32, 87.8], BAR_NAVY),
     ("GPU route", [1.29, 5.19, 20.7], BAR_NEUTRAL)]
_bars(dL, ["64K", "256K", "1M"], S, width=0.32, log=True)
dL.set_ylabel("added (ms)")
dL.axvline(1.5, color=SPINE_COLOR, linestyle="--", linewidth=1.2)
dL.text(0.60, 0.72, "gate ≤262K", transform=dL.transAxes, ha="right",
        va="top", fontsize=FONT_SIZE_BASE - 6, color=SPINE_COLOR)
S = [("SVE (ours)", [12.3, 33.3, 137.0], BAR_NAVY),
     ("Stock CPU", [21.6, 79.8, 4336.0], BAR_NEUTRAL)]
_bars(dR, ["64K", "256K", "1M"], S, width=0.32, log=True)
dR.set_ylabel("serial (ms)")
h, l = dL.get_legend_handles_labels()
h2, l2 = dR.get_legend_handles_labels()
band_legend(fig, h + h2, l + l2, 4)
save(fig, "m4-crossover", "m4b-grace-placement")

# ── m6 — tier tradeoff (2 panels; T1@512K measured OOM markers) ─────────────
fig, (eL, eR) = make_fig(2)
TIERS = [("T1 resident", BAR_NEUTRAL), ("T2 partial", BAR_NEUTRAL_CYAN),
         ("T3 streamed", BAR_NAVY)]
tok  = [[2062.7, None], [1524.3, 327.4], [698.4, 264.1]]
peak = [[14.05, None], [11.77, 172.4], [8.41, 117.65]]
OOM_RED = "#C0392B"
for ax, data, ylab in ((eL, tok, "tokens/s"), (eR, peak, "peak HBM (GiB)")):
    x = np.arange(2)
    ymax = max(v for row in data for v in row if v is not None)
    for i, ((name, color), vals) in enumerate(zip(TIERS, data)):
        off = (i - 1) * 0.24
        for gx, v in enumerate(vals):
            if v is None:  # measured OOM attempt
                h = 0.10 * ymax
                ax.bar(gx + off, h, 0.24, facecolor="none", edgecolor=OOM_RED,
                       linewidth=1.0, hatch="xx", zorder=3)
                ax.text(gx + off, h + 0.02 * ymax, "OOM", ha="center",
                        va="bottom", rotation=90, fontsize=FONT_SIZE_BASE - 6,
                        color=OOM_RED, fontweight="bold", zorder=4)
            else:
                ax.bar(gx + off, v, 0.24, color=color, edgecolor="white",
                       linewidth=0.8, zorder=3,
                       label=name if gx == 0 else None)
    ax.set_xticks(x, ["32K", "512K"])
    ax.set_xlim(*GROUP_XLIM)
    _style(ax)
    ax.set_ylabel(ylab)
h, l = eL.get_legend_handles_labels()
band_legend(fig, h, l, 3)
save(fig, "m6-tradeoff")

print("generated 5 figures @", FIGSIZE, "in")
