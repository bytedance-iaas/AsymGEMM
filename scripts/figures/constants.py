"""Shared style constants for every figure in scripts/figures/.

One place to manage fonts, font sizes, colors (the legend colors ARE the
stacked-bar segment colors), and per-figure size params, so all figures stay
visually consistent. Figure height and bar spacing live under FIGURE_PARAMS;
figure width is derived from the plotted x-span so wider plots naturally grow
when more bars are added.
"""
from __future__ import annotations

import glob
import os

import matplotlib
from matplotlib import font_manager

# --- Font family ----------------------------------------------------------
HELVETICA_LIKE_FONTS = [
    "Nimbus Sans",
    "Arial",
    "Helvetica",
]

# --- Font sizes (consistent across every figure) --------------------------
FONT_SIZE_BASE = 22         # general text / annotations
FONT_SIZE_TICK = 18         # x and y tick labels
FONT_SIZE_LEGEND = 20       # legend entries
FONT_SIZE_AXIS_TITLE = 24   # x and y axis labels
FONT_SIZE_SEGMENT = 18      # in-bar per-segment % labels
FONT_SIZE_MODEL = 20        # per-group model-name labels under the x-axis
# (no figure title anywhere)

# --- Colors: phase/segment -> color (legend reuses these exact colors) ----
PHASE_COLORS = {
    "Forward": "#5B8FB9",    # compute (light blue)
    "Backward": "#1F4E79",   # compute (dark blue)
    "Optimizer": "#E07B39",  # highlighted slice (orange)
}
PHASE_EDGE = {
    "Forward": "#ffffff",
    "Backward": "#ffffff",
    "Optimizer": "#7a3e16",
}
PHASE_HATCH = {
    "Optimizer": "////",
}

# Memory-breakdown segment colors (legend == stacked-bar colors)
MEMORY_COLORS = {
    "Model Weights": "#34495E",   # large fixed base (dark slate)
    "Optim. States": "#5B8FB9",   # LoRA weights+grads (blue sliver)
    "Activations": "#E07B39",     # grows with seq length (orange)
}
MEMORY_EDGE = {k: "#ffffff" for k in MEMORY_COLORS}

# Timeline colors reuse the existing paper palette where possible.
C2C_COLORS = {
    "ctc_rx": PHASE_COLORS["Forward"],
    "ctc_tx": "#B3261E",
    "reference": "#444444",
}
C2C_LABELS = {
    "ctc_rx": "C2C RX",
    "ctc_tx": "C2C TX",
}
UTIL_COLORS = {
    "gpu": C2C_COLORS["ctc_tx"],
    "cpu": C2C_COLORS["ctc_rx"],
}
UTIL_LABELS = {
    "gpu": "GPU",
    "cpu": "CPU",
}
TIMELINE_PHASE_COLORS = {
    "forward": "#5B8FB9",
    "backward": "#B3261E",
    "optimizer": "#D9A300",
}
TIMELINE_PHASE_LABELS = {
    "forward": "Forward",
    "backward": "Backward",
    "optimizer": "Optimizer",
}
TIMELINE_PHASE_ALPHA = 0.17

# --- Figure sizing --------------------------------------------------------
FIGURE_X_UNIT_INCHES = 9.2
FIGURE_SIDE_INCHES = 1.0


def xlim_from_positions(xpos, bar_width, x_margin):
    """Return x-limits that include full bar width plus edge padding."""
    x_pad = bar_width / 2.0 + x_margin
    return xpos[0] - x_pad, xpos[-1] + x_pad


def figure_size_for_xlim(xlim, height):
    """Keep height fixed; derive width from the actual plotted x-span."""
    x0, x1 = xlim
    return FIGURE_SIDE_INCHES + (x1 - x0) * FIGURE_X_UNIT_INCHES, height


# --- Per-figure size params (height/bar spacing differ per figure) --------
FIGURE_PARAMS = {
    "lora_timing": {
        "height": 4,
        "bar_width": 0.10,    # bar thickness only (independent)
        "pair_gap": 0.16,     # spacing of the 2 bars WITHIN a group
        "group_gap": 0.25,    # spacing BETWEEN groups (independent of pair_gap)
        "x_margin": 0.11,     # padding beyond the outermost bar edges
        "model_label_y": -0.15,  # model-name y position in x-axis coords
        "axes_left": 0.12,
        "axes_right": 0.985,
        "axes_bottom": 0.22,
        "axes_top": 0.86,
        "dpi": 200,
    },
    "memory": {
        "height": 4,
        "bar_width": 0.1,    # bar thickness only (independent)
        "pair_gap": 0.16,     # PyTorch vs SuperOffload within one model
        "group_gap": 0.25,    # spacing BETWEEN model groups
        "x_margin": 0.11,     # padding beyond the outermost bar edges
        "model_label_y": -0.15,  # model-name y position in x-axis coords
        "axes_left": 0.12,
        "axes_right": 0.985,
        "axes_bottom": 0.22,
        "axes_top": 0.86,
        "dpi": 200,
    },
    "timeline": {
        "width": 12,
        "c2c_width": 12,
        "height": 4,
        "axes_left": 0.12,
        "axes_right": 0.985,
        "axes_bottom": 0.22,
        "axes_top": 0.86,
        "dpi": 200,
        "line_width": 3,
        "grid_color": "#d1d5db",
        "boundary_color": "#9ca3af",
    },
}

# --- Horizontal legend ABOVE the axes box (not stacked inside) ------------
LEGEND_PARAMS = {
    "default": dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=1.4,
        columnspacing=1.8,
        handletextpad=0.6,
    ),
    "timeline": dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=1.5,
        columnspacing=1.2,
        handletextpad=0.6,
    ),
}


# user font dirs scanned so Helvetica-like faces (e.g. Nimbus Sans) render even
# when not installed system-wide; harmless if the dirs are empty/missing.
_USER_FONT_DIRS = [
    os.path.join(os.path.dirname(__file__), "fonts"),
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]
_FONTS_REGISTERED = False


def _register_user_fonts():
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    for d in _USER_FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.otf")) + glob.glob(os.path.join(d, "*.ttf")):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    _FONTS_REGISTERED = True


def apply_style():
    """Apply shared fonts + sizes globally via rcParams."""
    _register_user_fonts()
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        # DejaVu Sans is the always-present fallback so renders stay warning-free
        # when no Helvetica-like face is installed.
        "font.sans-serif": HELVETICA_LIKE_FONTS + ["DejaVu Sans"],
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_AXIS_TITLE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
    })


def horizontal_legend(ax, handles, legend_key="default"):
    """Add a single-row horizontal legend on top of the axes box."""
    if legend_key not in LEGEND_PARAMS:
        raise KeyError(f"unknown legend style: {legend_key}")
    kw = dict(LEGEND_PARAMS[legend_key])
    kw["ncol"] = len(handles)   # one horizontal row
    return ax.legend(handles=handles, **kw)


def layout(pair_gap, group_gap, n_groups):
    """Bar x-positions + group centers (2 bars per group). Fully orthogonal knobs:

    pair_gap  = spacing between the two bars WITHIN a group (center-to-center).
    group_gap = spacing BETWEEN groups, measured center-to-center of the two
                inner-facing bars -- independent of pair_gap AND bar_width.
    bar_width is NOT used here (it is the drawn width only), so it never moves a
    bar. Each knob changes exactly one thing: pair_gap spreads a pair without
    touching the between-group gap; group_gap moves groups apart without touching
    the pair; bar_width only thickens bars.
    """
    step = pair_gap + group_gap          # center-to-center between group centers
    centers = [g * step for g in range(n_groups)]
    xpos = [c + s for c in centers for s in (-pair_gap / 2.0, pair_gap / 2.0)]
    return xpos, centers
