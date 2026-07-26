#!/usr/bin/env python3
"""Motivation-section figures M2a/M2b/M3/M4/M6 from the banked measurement jsons.

Renders (PDF for the paper + PNG for visual QA) into scripts/figures/out/:
  m2a-kernel-case-study : adapter-GEMM case study (wall ms + analytic operand GB)
  m2b-scatter-fused     : scatter-fusion leg times (unfused vs fused)
  m3-composition        : per-module composition (bwd peak GiB / link GiB / bwd-window ms)
  m4-crossover          : Grace-kernel case study (added critical path + serial vs stock)
  m6-tradeoff           : tier tradeoff (tok/s + peak HBM; T1@512K = OOM marker)

Data: profiling_results/motivation/{m2a,m2b,m3,m4b,m4b_window,m6}.json, banked by
the bench scripts next to this file; NOTHING is re-measured here. Layouts follow
agent/impls/s04-p1-dgx-02-c06/motivation_v2_plots.md (MEASURED "FINAL FORM"
blocks override the original layouts). Bars = mean over the measured runs after
discarding each json's first run (warmup) when >= 3 runs were banked; thin
min-max caps span the kept runs. Fonts, colors, bar styling and the grouped-bar
layout helper all come from scripts/figures/constants.py (ep_balance house
style); the navy hero color marks our mechanism in every figure.
"""
# NB (2026-07-26): the PAPER's five motivation figures (m2a/m2b/m3/m4/m6)
# are emitted UNIFORM-SIZE by env/figures/plot_m2_row.py (Kevin's height
# standard, currently 9.9x4.125in). This script remains the data-faithful
# generator; if you regenerate from here, re-apply the standard before
# copying into the paper.

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# absolute() (not resolve()): scripts/figures is a symlink into env/, and
# resolving it would walk REPO_ROOT out of the repo checkout
REPO_ROOT = Path(__file__).absolute().parents[2]
FIGPKG = REPO_ROOT / "scripts" / "figures"
sys.path.insert(0, str(FIGPKG))
import constants as C  # noqa: E402

DATA_DIR = REPO_ROOT / "profiling_results" / "motivation"
OUT_DIR = FIGPKG / "out"

GIB = 1024.0 ** 3


# --- Shared protocol helpers ----------------------------------------------

def stats(vals):
    """mean/lo/hi after discarding the first run when >= 3 runs (protocol)."""
    vs = list(vals)
    if len(vs) >= 3:
        vs = vs[1:]
    return {"mean": sum(vs) / len(vs), "lo": min(vs), "hi": max(vs)}


def _load(name):
    return json.loads((DATA_DIR / name).read_text())


def _rows_label(n):
    k = n // 1024
    return f"{k // 1024}M" if k >= 1024 else f"{k}K"


def _fmt_val(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _nice_step(span, target_ticks=6.0):
    """Smallest 1/2/2.5/5 x 10^k step giving <= ~target_ticks gridlines."""
    raw = span / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


# --- Shared drawing helpers -----------------------------------------------

def _one_row(n_panels, xlim, pp):
    """Figure with n side-by-side panels; width derived from the x-span."""
    ax_w = (xlim[1] - xlim[0]) * C.FIGURE_X_UNIT_INCHES
    mid = pp.get("mid_in", 1.15) if n_panels > 1 else 0.0
    total_w = (pp["side_left_in"] + n_panels * ax_w
               + (n_panels - 1) * mid + pp["side_right_in"])
    total_h = pp["top_in"] + pp["ax_h_in"] + pp["bottom_in"]
    fig, axes = plt.subplots(1, n_panels, figsize=(total_w, total_h),
                             dpi=pp["dpi"], squeeze=False)
    fig.subplots_adjust(left=pp["side_left_in"] / total_w,
                        right=1.0 - pp["side_right_in"] / total_w,
                        bottom=pp["bottom_in"] / total_h,
                        top=1.0 - pp["top_in"] / total_h,
                        wspace=(mid / ax_w) if n_panels > 1 else 0.0)
    return fig, list(axes[0]), total_w, total_h


def _fig_legend(fig, handles, total_h, pp):
    """One horizontal legend row 0.1in above the axes tops (house style)."""
    kw = dict(C.LEGEND_PARAMS["default"])
    kw["ncol"] = len(handles)
    kw["bbox_to_anchor"] = (0.5, 1.0 - pp["top_in"] / total_h + 0.10 / total_h)
    fig.legend(handles=handles, **kw)


def _style_axis(ax):
    ax.grid(axis="y", color=C.GRID_COLOR, linewidth=0.9, linestyle=(0, (4, 3)),
            alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(C.SPINE_COLOR)
        spine.set_linewidth(1.2)


def _linear_yticks(ax, y_top):
    step = _nice_step(y_top)
    ticks = [i * step for i in range(int(y_top / step) + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(0, y_top)


def _group_xticks(ax, centers, labels, xlim):
    ax.set_xlim(*xlim)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)


def _bar(ax, x, mean, bar_w, color):
    ax.bar(x, mean, width=bar_w, color=color, edgecolor="#ffffff",
           linewidth=0.8, zorder=3)


def _caps(ax, xs, means, e_lo, e_hi):
    """Thin min-max caps over the kept runs (house style)."""
    ax.errorbar(xs, means, yerr=[e_lo, e_hi], fmt="none", ecolor=C.SPINE_COLOR,
                elinewidth=1.0, capsize=2.6, capthick=1.0, zorder=4)


def _rot_label(ax, x, y, text, fontsize=None):
    """Rotated value label above a bar/cap (ep_balance label style)."""
    ax.text(x, y, text, ha="center", va="bottom", rotation=90,
            fontsize=fontsize or (C.FONT_SIZE_SEGMENT - 4),
            color=C.SUBTLE_TEXT, zorder=4)


def _save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):   # PDF for the paper, PNG for quick visual QA
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


# --- M2a: adapter-GEMM case study -----------------------------------------

M2A_PARAMS = dict(bar_width=0.042, intra_gap=0.007, group_gap=0.055,
                  x_margin=0.032, side_left_in=1.15, side_right_in=0.18,
                  mid_in=1.25, ax_h_in=3.9, top_in=0.95, bottom_in=0.66,
                  dpi=200)
M2A_SERIES = [
    {"key": "resident", "label": "Resident",        "color": C.BAR_NEUTRAL},
    {"key": "staged",   "label": "Staged",          "color": C.BAR_NEUTRAL_CYAN},
    {"key": "streamed", "label": "Streamed (ours)", "color": C.BAR_NAVY},
]


def fig_m2a():
    """2 panels (wall ms | analytic operand GB) x 2 row-groups x 3 executions."""
    d = _load("m2a.json")
    pp = M2A_PARAMS
    groups = []
    for sh in d["shapes"]:
        t = {s["key"]: stats([r[f'{s["key"]}_ms'] for r in sh["runs"]])
             for s in M2A_SERIES}
        groups.append({"label": f'{_rows_label(sh["rows"])} rows',
                       "time": t, "mem": sh["mem_gb"]})

    print("[m2a] parsed (ms | GB):")
    for g in groups:
        row = "  ".join(f'{s["key"]}={g["time"][s["key"]]["mean"]:8.3f}/'
                        f'{g["mem"][s["key"]]:.5f}' for s in M2A_SERIES)
        print(f'  {g["label"]:10s} {row}')

    bars, centers = C.grouped_layout(len(groups), len(M2A_SERIES),
                                     pp["bar_width"], pp["intra_gap"],
                                     pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, pp["bar_width"], pp["x_margin"])
    fig, (ax_t, ax_m), total_w, total_h = _one_row(2, xlim, pp)

    # Panel L: wall time, with per-bar value labels (the Resident slivers are
    # the 30x residency premium -- unreadable without the number).
    ymax = max(g["time"][s["key"]]["hi"] for g in groups for s in M2A_SERIES)
    y_top = ymax * 1.30                       # headroom for rotated labels
    pad = 0.02 * y_top
    xs, means, e_lo, e_hi = [], [], [], []
    for g, grp in enumerate(groups):
        for s, ser in enumerate(M2A_SERIES):
            st = grp["time"][ser["key"]]
            x = bars[g][s]
            _bar(ax_t, x, st["mean"], pp["bar_width"], ser["color"])
            _rot_label(ax_t, x, st["hi"] + pad, _fmt_val(st["mean"]))
            xs.append(x)
            means.append(st["mean"])
            e_lo.append(st["mean"] - st["lo"])
            e_hi.append(st["hi"] - st["mean"])
    _caps(ax_t, xs, means, e_lo, e_hi)
    _style_axis(ax_t)
    _linear_yticks(ax_t, y_top)
    _group_xticks(ax_t, centers, [g["label"] for g in groups], xlim)
    ax_t.set_ylabel("Wall time (ms)")

    # Panel R: analytic GPU operand bytes (no runs -> no caps). Streamed holds
    # only A (~0.7 MB): invisible at GB scale, so the label carries it.
    mmax = max(g["mem"][s["key"]] for g in groups for s in M2A_SERIES)
    m_top = mmax * 1.30
    mpad = 0.02 * m_top
    for g, grp in enumerate(groups):
        for s, ser in enumerate(M2A_SERIES):
            v = grp["mem"][ser["key"]]
            x = bars[g][s]
            _bar(ax_m, x, v, pp["bar_width"], ser["color"])
            label = f"{v * 1e3:.1f} MB" if v < 0.01 else _fmt_val(v)
            _rot_label(ax_m, x, v + mpad, label)
    _style_axis(ax_m)
    _linear_yticks(ax_m, m_top)
    _group_xticks(ax_m, centers, [g["label"] for g in groups], xlim)
    ax_m.set_ylabel("Operand memory (GB)")

    handles = [Patch(facecolor=s["color"], edgecolor="#ffffff",
                     label=s["label"]) for s in M2A_SERIES]
    _fig_legend(fig, handles, total_h, pp)
    _save(fig, "m2a-kernel-case-study")


# --- M2b: scatter fusion ---------------------------------------------------

M2B_PARAMS = dict(bar_width=0.05, intra_gap=0.008, group_gap=0.09,
                  x_margin=0.04, side_left_in=1.15, side_right_in=0.18,
                  ax_h_in=3.9, top_in=0.95, bottom_in=1.0, dpi=200)
M2B_SERIES = [
    {"key": "unfused", "label": "Unfused", "color": C.BAR_NEUTRAL},
    {"key": "fused",   "label": "Fused",   "color": C.BAR_NAVY},
]
M2B_LEGS = [("fwd", "Forward"), ("bwd", "Gate/up\ngradient")]


def fig_m2b():
    """ONE panel: leg wall time; 2 legs x 2 executions. No annotations."""
    d = _load("m2b.json")
    pp = M2B_PARAMS
    t = {leg: {s["key"]: stats([r[f'{leg}_{s["key"]}_ms'] for r in d["runs"]])
               for s in M2B_SERIES} for leg, _ in M2B_LEGS}

    print("[m2b] parsed (ms):")
    for leg, name in M2B_LEGS:
        row = "  ".join(f'{s["key"]}={t[leg][s["key"]]["mean"]:6.2f}'
                        for s in M2B_SERIES)
        print(f"  {leg}: {row}")

    bars, centers = C.grouped_layout(len(M2B_LEGS), len(M2B_SERIES),
                                     pp["bar_width"], pp["intra_gap"],
                                     pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, pp["bar_width"], pp["x_margin"])
    fig, (ax,), total_w, total_h = _one_row(1, xlim, pp)

    ymax = max(t[leg][s["key"]]["hi"] for leg, _ in M2B_LEGS for s in M2B_SERIES)
    y_top = ymax * 1.15
    xs, means, e_lo, e_hi = [], [], [], []
    for g, (leg, _) in enumerate(M2B_LEGS):
        for s, ser in enumerate(M2B_SERIES):
            st = t[leg][ser["key"]]
            _bar(ax, bars[g][s], st["mean"], pp["bar_width"], ser["color"])
            xs.append(bars[g][s])
            means.append(st["mean"])
            e_lo.append(st["mean"] - st["lo"])
            e_hi.append(st["hi"] - st["mean"])
    _caps(ax, xs, means, e_lo, e_hi)
    _style_axis(ax)
    _linear_yticks(ax, y_top)
    _group_xticks(ax, centers, [name for _, name in M2B_LEGS], xlim)
    ax.set_ylabel("Wall time (ms)")

    handles = [Patch(facecolor=s["color"], edgecolor="#ffffff",
                     label=s["label"]) for s in M2B_SERIES]
    _fig_legend(fig, handles, total_h, pp)
    _save(fig, "m2b-scatter-fused")


# --- M3: per-module composition bench -------------------------------------

M3_PARAMS = dict(bar_width=0.038, intra_gap=0.006, group_gap=0.05,
                 x_margin=0.028, side_left_in=1.15, side_right_in=0.18,
                 mid_in=1.80, ax_h_in=3.9, top_in=0.95, bottom_in=0.66,
                 dpi=200)   # mid widened: panel 3's two-line ylabel + 4-digit ticks
M3_SERIES = [
    {"key": "recompute_all", "label": "Recompute-all",  "color": C.BAR_NEUTRAL},
    {"key": "offload_all",   "label": "Offload-all",    "color": C.BAR_NEUTRAL_CYAN},
    {"key": "composed",      "label": "Composed (ours)", "color": C.BAR_NAVY},
]
M3_MODULES = [("attention", "Attention"), ("mlp", "MLP")]


def _m3_metrics(module_rec):
    """Per-rep metric lists (stats() drops the warmup rep, matching agg)."""
    reps = module_rec["reps"]
    link = [(r["fwd"]["link"]["d2h_bytes"] + r["fwd"]["link"]["h2d_bytes"]
             + r["bwd"]["link"]["d2h_bytes"] + r["bwd"]["link"]["h2d_bytes"])
            / GIB for r in reps]
    return {
        "fwd_ms": stats([r["fwd"]["ev_ms"] for r in reps]),
        "bwd_ms": stats([r["bwd"]["ev_ms"] for r in reps]),
        "bwd_peak_gib": stats([r["bwd"]["peak_bytes"] / GIB for r in reps]),
        "link_gib": stats(link),
    }


def fig_m3():
    """3 panels (bwd peak GiB | link GiB | bwd-window ms) x 2 modules x 3
    policies. The time panel reports the backward window only: that is where
    the composition acts, and the forward segment measured policy-independent
    (the parsed print keeps the fwd means for the record)."""
    d = _load("m3.json")
    pp = M3_PARAMS
    vals = {mod: {s["key"]: _m3_metrics(d["policies"][s["key"]]["modules"][mod])
                  for s in M3_SERIES} for mod, _ in M3_MODULES}

    print("[m3] parsed (fwd ms / bwd ms / bwd peak GiB / link GiB):")
    for mod, _ in M3_MODULES:
        for s in M3_SERIES:
            v = vals[mod][s["key"]]
            print(f'  {mod:9s} {s["key"]:13s} {v["fwd_ms"]["mean"]:6.0f} '
                  f'{v["bwd_ms"]["mean"]:6.0f} {v["bwd_peak_gib"]["mean"]:5.1f} '
                  f'{v["link_gib"]["mean"]:5.1f}')

    bars, centers = C.grouped_layout(len(M3_MODULES), len(M3_SERIES),
                                     pp["bar_width"], pp["intra_gap"],
                                     pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, pp["bar_width"], pp["x_margin"])
    fig, (ax_p, ax_l, ax_t), total_w, total_h = _one_row(3, xlim, pp)
    mod_labels = [name for _, name in M3_MODULES]

    # Panel 1: peak GPU memory during the backward window.
    def flat_panel(ax, metric, ylabel, zero_label=False):
        ymax = max(vals[m][s["key"]][metric]["hi"]
                   for m, _ in M3_MODULES for s in M3_SERIES)
        y_top = ymax * 1.15
        xs, means, e_lo, e_hi = [], [], [], []
        for g, (mod, _) in enumerate(M3_MODULES):
            for s, ser in enumerate(M3_SERIES):
                st = vals[mod][ser["key"]][metric]
                _bar(ax, bars[g][s], st["mean"], pp["bar_width"], ser["color"])
                if zero_label and st["mean"] == 0:
                    ax.text(bars[g][s], 0.012 * y_top, "0", ha="center",
                            va="bottom", fontsize=C.FONT_SIZE_SEGMENT - 4,
                            color=C.SUBTLE_TEXT, zorder=4)
                xs.append(bars[g][s])
                means.append(st["mean"])
                e_lo.append(st["mean"] - st["lo"])
                e_hi.append(st["hi"] - st["mean"])
        _caps(ax, xs, means, e_lo, e_hi)
        _style_axis(ax)
        _linear_yticks(ax, y_top)
        _group_xticks(ax, centers, mod_labels, xlim)
        ax.set_ylabel(ylabel)

    flat_panel(ax_p, "bwd_peak_gib", "Peak HBM (GiB)")
    flat_panel(ax_l, "link_gib", "Link traffic (GiB)", zero_label=True)

    # Panel 3: backward-window time only (the composition acts there; the
    # forward segment is policy-independent and stays out of the figure).
    # Two-line ylabel: the single line overruns the 3.9-in axis height.
    flat_panel(ax_t, "bwd_ms", "Backward-window\ntime (ms)")

    handles = [Patch(facecolor=s["color"], edgecolor="#ffffff",
                     label=s["label"]) for s in M3_SERIES]
    _fig_legend(fig, handles, total_h, pp)
    _save(fig, "m3-composition")


# --- M4: Grace-kernel case study (crossover) ------------------------------

M4_PARAMS = dict(bar_width=0.044, intra_gap=0.007, group_gap=0.05,
                 x_margin=0.032, side_left_in=1.15, side_right_in=0.18,
                 mid_in=1.55, ax_h_in=3.9, top_in=0.95, bottom_in=1.12,
                 panel_label_y=-0.14, dpi=200)
# color follows the entity across both panels: ours=navy, GPU route=cyan,
# stock CPU=gray -- one shared legend row covers the whole figure
M4_CPU = {"label": "CPU kernel (ours)", "color": C.BAR_NAVY}
M4_GPU = {"label": "GPU route", "color": C.BAR_NEUTRAL_CYAN}
M4_STOCK = {"label": "Stock CPU", "color": C.BAR_NEUTRAL}
M4_ARMING_ROWS = "≈262K rows"      # shipped arming ceiling


def _log_panel(ax, y_lim, ticks):
    ax.set_yscale("log")
    _style_axis(ax)
    ax.set_ylim(*y_lim)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def fig_m4():
    """(a) added critical path under a busy backward; (b) serial ours-vs-stock."""
    dw = _load("m4b_window.json")
    ds = _load("m4b.json")
    pp = M4_PARAMS

    win = [{"rows": c["rows"],
            "cpu": stats([r["cpu_added_ms"] for r in c["runs"]]),
            "gpu": stats([r["gpu_added_ms"] for r in c["runs"]])}
           for c in dw["clusters"]]
    ser = [{"rows": c["rows"],
            "ours": stats([r["cpu_ours_ms"] for r in c["runs"]]),
            "stock": stats([r["cpu_stock_ms"] for r in c["runs"]])}
           for c in ds["deposit"]]

    print("[m4] parsed (a: added ms cpu/gpu; b: serial ms ours/stock):")
    for w, s in zip(win, ser):
        print(f'  rows={_rows_label(w["rows"]):>4s}: '
              f'+{w["cpu"]["mean"]:6.2f}/+{w["gpu"]["mean"]:6.2f} | '
              f'{s["ours"]["mean"]:7.1f}/{s["stock"]["mean"]:7.1f}')

    bars, centers = C.grouped_layout(len(win), 2, pp["bar_width"],
                                     pp["intra_gap"], pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, pp["bar_width"], pp["x_margin"])
    fig, (ax_a, ax_b), total_w, total_h = _one_row(2, xlim, pp)
    row_labels = [_rows_label(w["rows"]) for w in win]

    def draw(ax, recs, keys, styles):
        xs, means, e_lo, e_hi = [], [], [], []
        for g, rec in enumerate(recs):
            for s, key in enumerate(keys):
                st = rec[key]
                _bar(ax, bars[g][s], st["mean"], pp["bar_width"],
                     styles[s]["color"])
                xs.append(bars[g][s])
                means.append(st["mean"])
                e_lo.append(st["mean"] - st["lo"])
                e_hi.append(st["hi"] - st["mean"])
        _caps(ax, xs, means, e_lo, e_hi)

    # Panel (a): added-critical-path; log y keeps the 1.3 ms parity groups and
    # the 88 ms overflow group readable on one axis.
    draw(ax_a, win, ("cpu", "gpu"), (M4_CPU, M4_GPU))
    _log_panel(ax_a, (0.5, 400), [1, 10, 100])
    _group_xticks(ax_a, centers, row_labels, xlim)
    ax_a.set_ylabel("Added makespan (ms)")
    ax_a.text(0.5, pp["panel_label_y"], "During busy backward",
              transform=ax_a.transAxes, ha="center", va="top",
              fontsize=C.FONT_SIZE_MODEL, fontweight="bold", clip_on=False)
    # dashed crossover marker at the shipped arming ceiling (between 256K, 1M);
    # the line stops below its label instead of striking through it
    x_cross = (centers[1] + centers[2]) / 2.0
    ax_a.axvline(x_cross, ymax=0.80, color=C.ACCENT_RED,
                 linestyle=(0, (5, 4)), linewidth=1.6, zorder=2)
    ax_a.text(x_cross, 0.965, "arming ceiling\n" + M4_ARMING_ROWS,
              transform=blended_transform_factory(ax_a.transData,
                                                  ax_a.transAxes),
              ha="center", va="top", fontsize=C.FONT_SIZE_SEGMENT - 4,
              color=C.ACCENT_RED, linespacing=1.15, zorder=4)

    # Panel (b): serial ours-vs-stock; log y since stock hits 4336 ms at 1M.
    draw(ax_b, ser, ("ours", "stock"), (M4_CPU, M4_STOCK))
    _log_panel(ax_b, (5, 8000), [10, 100, 1000])
    _group_xticks(ax_b, centers, row_labels, xlim)
    ax_b.set_ylabel("Wall time (ms)")
    ax_b.text(0.5, pp["panel_label_y"], "Serial (kernel alone)",
              transform=ax_b.transAxes, ha="center", va="top",
              fontsize=C.FONT_SIZE_MODEL, fontweight="bold", clip_on=False)

    handles = [Patch(facecolor=s["color"], edgecolor="#ffffff",
                     label=s["label"]) for s in (M4_CPU, M4_GPU, M4_STOCK)]
    _fig_legend(fig, handles, total_h, pp)
    _save(fig, "m4-crossover")


# --- M6: tier tradeoff (throughput vs peak HBM) ----------------------------

M6_PARAMS = dict(bar_width=0.042, intra_gap=0.007, group_gap=0.055,
                 x_margin=0.032, side_left_in=1.50, side_right_in=0.18,
                 mid_in=1.25, ax_h_in=3.9, top_in=0.95, bottom_in=0.66,
                 dpi=200)   # left widened for the 4-digit tok/s ticks
M6_SERIES = [
    {"key": "T1", "label": "T1 resident", "color": C.BAR_NEUTRAL},
    {"key": "T2", "label": "T2 partial",  "color": C.BAR_NEUTRAL_CYAN},
    {"key": "T3", "label": "T3 streamed", "color": C.BAR_NAVY},
]
M6_SEQS = [32000, 512000]


def _oom_marker(ax, x, bar_w, y_top):
    """NO bar: red cross-hatched open box + bold OOM annotation (plot_main's
    OOM text style; the caption explains the marker)."""
    h = 0.10 * y_top
    ax.bar(x, h, width=bar_w, facecolor="none", edgecolor=C.MEMSAVE_OOM_COLOR,
           linewidth=1.0, hatch="xx", zorder=3)
    ax.text(x, h + 0.02 * y_top, "OOM", ha="center", va="bottom", rotation=90,
            fontsize=C.FONT_SIZE_SEGMENT - 4, color=C.MEMSAVE_OOM_COLOR,
            fontweight="bold", zorder=4)


def fig_m6():
    """2 panels (tok/s | peak HBM GiB) x 2 length-groups x 3 tiers; T1@512K
    drew a measured OOM (attempted run), so both panels get the marker."""
    d = _load("m6.json")
    pp = M6_PARAMS
    cells = {(c["tier"], c["seq_len"]): c for c in d["cells"]}

    print("[m6] parsed (tok/s | peak alloc GiB):")
    for seq in M6_SEQS:
        row = "  ".join(
            f'{s["key"]}=   OOM' if cells[(s["key"], seq)]["status"] == "oom"
            else f'{s["key"]}={cells[(s["key"], seq)]["tokens_per_s"]:6.1f}/'
                 f'{cells[(s["key"], seq)]["peak_gpu_allocated_gib"]:6.2f}'
            for s in M6_SERIES)
        print(f"  {seq // 1000:>3d}K  {row}")

    bars, centers = C.grouped_layout(len(M6_SEQS), len(M6_SERIES),
                                     pp["bar_width"], pp["intra_gap"],
                                     pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, pp["bar_width"], pp["x_margin"])
    fig, (ax_t, ax_m), total_w, total_h = _one_row(2, xlim, pp)
    seq_labels = [f"{seq // 1000}K" for seq in M6_SEQS]
    meas = [c for c in cells.values() if c["status"] == "measured"]

    # Panel L: throughput; bar = the banked mean-step tok/s, thin caps span
    # the per-step tok/s of the kept steps (w1+m2 protocol).
    ymax = max(max(c["batch"] * c["seq_len"] / t
                   for t in c["measured_step_seconds"]) for c in meas)
    y_top = ymax * 1.30                       # headroom for rotated labels
    pad = 0.02 * y_top
    xs, means, e_lo, e_hi = [], [], [], []
    for g, seq in enumerate(M6_SEQS):
        for s, ser in enumerate(M6_SERIES):
            c = cells[(ser["key"], seq)]
            x = bars[g][s]
            if c["status"] == "oom":
                _oom_marker(ax_t, x, pp["bar_width"], y_top)
                continue
            per_step = [c["batch"] * c["seq_len"] / t
                        for t in c["measured_step_seconds"]]
            v = c["tokens_per_s"]
            _bar(ax_t, x, v, pp["bar_width"], ser["color"])
            _rot_label(ax_t, x, max(per_step) + pad, _fmt_val(v))
            xs.append(x)
            means.append(v)
            e_lo.append(v - min(per_step))
            e_hi.append(max(per_step) - v)
    _caps(ax_t, xs, means, e_lo, e_hi)
    _style_axis(ax_t)
    _linear_yticks(ax_t, y_top)
    _group_xticks(ax_t, centers, seq_labels, xlim)
    ax_t.set_ylabel("Throughput (tok/s)")

    # Panel R: allocator peak (one number per run -> no caps); the 32K bars
    # are slivers on the 512K scale, so the labels carry them.
    mmax = max(c["peak_gpu_allocated_gib"] for c in meas)
    m_top = mmax * 1.30
    mpad = 0.02 * m_top
    for g, seq in enumerate(M6_SEQS):
        for s, ser in enumerate(M6_SERIES):
            c = cells[(ser["key"], seq)]
            x = bars[g][s]
            if c["status"] == "oom":
                _oom_marker(ax_m, x, pp["bar_width"], m_top)
                continue
            v = c["peak_gpu_allocated_gib"]
            _bar(ax_m, x, v, pp["bar_width"], ser["color"])
            _rot_label(ax_m, x, v + mpad, _fmt_val(v))
    _style_axis(ax_m)
    _linear_yticks(ax_m, m_top)
    _group_xticks(ax_m, centers, seq_labels, xlim)
    ax_m.set_ylabel("Peak HBM (GiB)")

    handles = [Patch(facecolor=s["color"], edgecolor="#ffffff",
                     label=s["label"]) for s in M6_SERIES]
    _fig_legend(fig, handles, total_h, pp)
    _save(fig, "m6-tradeoff")


# --- Entry point -----------------------------------------------------------

FIGURES = {"m2a": fig_m2a, "m2b": fig_m2b, "m3": fig_m3, "m4": fig_m4,
           "m6": fig_m6}


def main():
    ap = argparse.ArgumentParser(description="Render the motivation figures.")
    ap.add_argument("--only", choices=sorted(FIGURES), default=None,
                    help="render just one figure (default: all)")
    args = ap.parse_args()
    C.apply_style()
    for name, fn in FIGURES.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
