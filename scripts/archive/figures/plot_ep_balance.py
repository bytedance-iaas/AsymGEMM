#!/usr/bin/env python3
"""EP-effectiveness figures: one two-panel figure per MoE model.

Each figure has two side-by-side panels (Expert GEMM | MoE Block) sharing one
horizontal legend on top. Per panel: four z-skew clusters (z=0.5..2.0), four
bars per cluster -- EP (owned) and sDP as muted baselines, our sEP plan/queue
as the navy hero + saturated teal. Bar height = mean wall ms over the 3 seeded
shuffles; a thin cap spans min..max. Rotated labels above the bars (color =
reference, keyed in the left panel): gray on EP = its mean GPU-imbalance %
(what our balancing removes); red = reduction vs EP (on sDP and both sEP
bars); dark blue = reduction vs sDP (second column on the sEP bars).
Arrows are sign-aware: ↓ faster, ↑ slower, 0% = under half a percent.

Data is the banked ep_balance_bench sweep under profiling_results/profiling_both_skew/ (see the
per-model json pairs in MODELS; archive_*/ subdirs hold stale copies). Nothing
is re-measured here. Shared fonts/sizes/colors/figure params + the grouped-bar
layout helper all come from constants.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import constants as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "profiling_results/profiling_both_skew"

# --- Banked benchmark files (top-level ONLY; archive_*/ are stale) ---------
MODELS = [
    {"key": "q3-30b-a3b", "name": "Qwen3-30B-A3B", "stem": "ep_balance_q330b",
     "gemm": "table1_micro.json", "moe": "table1c_moe.json"},
    {"key": "q3-235b-a22b", "name": "Qwen3-235B-A22B", "stem": "ep_balance_q3235b",
     "gemm": "table1_q3235b_gemm.json", "moe": "table1_q3235b_moe.json"},
    {"key": "q35-122b-a10b", "name": "Qwen3.5-122B-A10B", "stem": "ep_balance_q35122b",
     "gemm": "table1_q35122b_gemm.json", "moe": "table1_q35122b_moe.json"},
    {"key": "l4-scout", "name": "Llama-4-Scout", "stem": "ep_balance_l4scout",
     "gemm": "table1_l4scout_gemm.json", "moe": "table1_l4scout_moe.json"},
]

# Bar series, left -> right within a cluster (muted baselines, then ours).
SERIES = [
    {"key": "owned", "label": "EP",          "color": C.BAR_NEUTRAL},
    {"key": "sdp",   "label": "sDP",         "color": C.BAR_NEUTRAL_CYAN},
    {"key": "plan",  "label": "sEP (plan)",  "color": C.BAR_NAVY},
    {"key": "queue", "label": "sEP (queue)", "color": C.BAR_TEAL},
]
ZS = ["0.5", "1.0", "1.5", "2.0"]          # zipf skew sweep (3 seeds each)
PANELS = [("gemm", "Expert GEMM"), ("moe", "MoE Block")]

_PARAMS = C.FIGURE_PARAMS["ep_balance"]


def load_scope(path):
    """-> {z: {mode: {mean, lo, hi} (ms), "imb": mean EP imbalance 0..1}}."""
    data = json.loads(Path(path).read_text())
    out = {}
    for z in ZS:
        # exact-name selection; skips zipf0.0/zipf0.8/worst:/median: cases
        sel = [c for c in data["cases"]
               if c["case"] in {f"zipf{z}|seed{n}" for n in range(3)}]
        if len(sel) != 3:
            raise ValueError(f"{path}: expected 3 seeds for zipf{z}, got {len(sel)}")
        per = {}
        for s in SERIES:
            ms = [c[s["key"]]["wall_s"] * 1e3 for c in sel]
            per[s["key"]] = {"mean": sum(ms) / 3, "lo": min(ms), "hi": max(ms)}
        per["imb"] = sum(c["owned"]["imbalance"] for c in sel) / 3
        out[z] = per
    return out


def _nice_step(ymax, target_ticks=5.5):
    """Smallest 1/2/2.5/5 x 10^k step giving <= ~target_ticks gridlines."""
    raw = ymax / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _fmt_pct(ref, val):
    """Sign-aware reduction label vs a reference mean: ↓12%, ↑3%, or 0%."""
    pct = (ref - val) / ref * 100.0
    if abs(pct) < 0.5:
        return "0%"
    return f"{'↓' if pct > 0 else '↑'}{abs(pct):.0f}%"


def _draw_panel(ax, per_z, caption, bars, centers, xlim, show_key):
    pp = _PARAMS
    bar_w = pp["bar_width"]
    ymax = max(per_z[z][s["key"]]["hi"] for z in ZS for s in SERIES)
    y_top = ymax * 1.20   # headroom for error caps + rotated % labels
    pad = 0.014 * y_top   # gap between an error cap and its label base
    dx = 0.010            # half-offset of the two label columns on sEP bars

    # rotated labels: a column is ~one glyph tall, so neighboring bars' labels
    # never collide even when bar tops are equal
    def label(x, y, text, color, weight="bold"):
        ax.text(x, y, text, ha="center", va="bottom", rotation=90,
                fontsize=C.FONT_SIZE_SEGMENT - 6, color=color,
                fontweight=weight, zorder=4)

    xs, means, e_lo, e_hi = [], [], [], []
    for g, z in enumerate(ZS):
        for s, series in enumerate(SERIES):
            v = per_z[z][series["key"]]
            x = bars[g][s]
            ax.bar(x, v["mean"], width=bar_w, color=series["color"],
                   edgecolor="#ffffff", linewidth=0.8, zorder=3)
            xs.append(x)
            means.append(v["mean"])
            e_lo.append(v["mean"] - v["lo"])
            e_hi.append(v["hi"] - v["mean"])
        ep = per_z[z]["owned"]
        sdp = per_z[z]["sdp"]
        # EP bar: its mean GPU-imbalance % (the cost our balancing removes)
        label(bars[g][0], ep["hi"] + pad, f"{per_z[z]['imb'] * 100:.0f}%",
              C.SUBTLE_TEXT, weight="normal")
        # sDP bar: reduction vs EP
        label(bars[g][1], sdp["hi"] + pad, _fmt_pct(ep["mean"], sdp["mean"]),
              C.ACCENT_RED)
        # sEP bars: two label columns -- vs EP (left, red), vs sDP (right, blue)
        for s in (2, 3):
            v = per_z[z][SERIES[s]["key"]]
            label(bars[g][s] - dx, v["hi"] + pad,
                  _fmt_pct(ep["mean"], v["mean"]), C.ACCENT_RED)
            label(bars[g][s] + dx, v["hi"] + pad,
                  _fmt_pct(sdp["mean"], v["mean"]), C.DARK_BLUE)

    if show_key:   # color key for the % labels (once, on the left panel)
        ax.text(0.025, 0.955, "↓% vs EP", transform=ax.transAxes, ha="left",
                va="top", fontsize=C.FONT_SIZE_SEGMENT - 6, color=C.ACCENT_RED,
                fontweight="bold", zorder=4)
        ax.text(0.025, 0.885, "↓% vs sDP", transform=ax.transAxes, ha="left",
                va="top", fontsize=C.FONT_SIZE_SEGMENT - 6, color=C.DARK_BLUE,
                fontweight="bold", zorder=4)

    # thin min..max caps over the 3 seeds
    ax.errorbar(xs, means, yerr=[e_lo, e_hi], fmt="none", ecolor=C.SPINE_COLOR,
                elinewidth=1.0, capsize=2.6, capthick=1.0, zorder=4)

    ax.set_ylim(0, y_top)
    step = _nice_step(ymax)
    n_ticks = int(y_top / step)
    ticks = [i * step for i in range(n_ticks + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(*xlim)

    ax.set_xticks(centers)
    ax.set_xticklabels([f"z={z}" for z in ZS])
    ax.tick_params(axis="x", length=0)
    ax.text(0.5, pp["panel_label_y"], caption, transform=ax.transAxes,
            ha="center", va="top", fontsize=C.FONT_SIZE_MODEL,
            fontweight="bold", clip_on=False)

    ax.set_ylabel("Wall time (ms)")
    ax.grid(axis="y", color=C.GRID_COLOR, linewidth=0.9, linestyle=(0, (4, 3)),
            alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(C.SPINE_COLOR)
        spine.set_linewidth(1.2)


def build_figure(model):
    C.apply_style()
    pp = _PARAMS
    bar_w = pp["bar_width"]

    scopes = {scope: load_scope(DATA_DIR / model[scope]) for scope, _ in PANELS}
    _print_parsed(model, scopes)

    bars, centers = C.grouped_layout(
        len(ZS), len(SERIES), bar_w, pp["intra_gap"], pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, bar_w, pp["x_margin"])

    # two equal panels: per-panel width from the shared x-span, plus a little
    # extra so panel B's y-label fits in the mid gap
    panel_w, height = C.figure_size_for_xlim(xlim, pp["height"])
    total_w = 2 * panel_w + pp["mid_extra_in"]
    ax_w = (xlim[1] - xlim[0]) * C.FIGURE_X_UNIT_INCHES
    left = pp["side_left_in"] / total_w
    right = 1.0 - pp["side_right_in"] / total_w
    wspace = (total_w - pp["side_left_in"] - pp["side_right_in"] - 2 * ax_w) / ax_w

    fig, axes = plt.subplots(1, 2, figsize=(total_w, height), dpi=pp["dpi"])
    fig.subplots_adjust(left=left, right=right, bottom=pp["axes_bottom"],
                        top=pp["axes_top"], wspace=wspace)

    for ax, (scope, caption) in zip(axes, PANELS):
        _draw_panel(ax, scopes[scope], caption, bars, centers, xlim,
                    show_key=(scope == PANELS[0][0]))

    # model name on the caption line, centered in the mid gap (never a title)
    y_name = pp["axes_bottom"] + pp["panel_label_y"] * (pp["axes_top"] - pp["axes_bottom"])
    fig.text(0.5, y_name, model["name"], ha="center", va="top",
             fontsize=C.FONT_SIZE_SEGMENT, color=C.SUBTLE_TEXT)

    handles = [Patch(facecolor=s["color"], edgecolor="#ffffff", label=s["label"])
               for s in SERIES]
    kw = dict(C.LEGEND_PARAMS["default"])
    kw["ncol"] = len(handles)
    kw["bbox_to_anchor"] = (0.5, pp["legend_y"])
    fig.legend(handles=handles, **kw)
    return fig


def _print_parsed(model, scopes):
    """Spot-check table: compare against the ANSWER KEY in agent/plot_ep.md."""
    print(f"[{model['key']}]")
    for scope, caption in PANELS:
        for z in ZS:
            per = scopes[scope][z]
            row = "  ".join(f"{s['key']}={per[s['key']]['mean']:6.1f}" for s in SERIES)
            print(f"  {caption:11s} z{z}: {row}  imb={per['imb'] * 100:3.0f}%")


def main():
    ap = argparse.ArgumentParser(description="Plot the per-model EP-effectiveness figures.")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--model", choices=[m["key"] for m in MODELS] + ["all"], default="all")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        if args.model not in ("all", model["key"]):
            continue
        fig = build_figure(model)
        for ext in ("pdf", "png"):   # PDF for the paper, PNG for quick visual QA
            out_path = out_dir / f"{model['stem']}.{ext}"
            fig.savefig(out_path, dpi=_PARAMS["dpi"])
            print(f"wrote {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
