#!/usr/bin/env python3
"""Main memory-saving figure(s) for AsymLoRA.

Grouped vertical bars (one group per model, one bar per system) mirroring the
reference throughput figure's format/style: a horizontal legend of systems on
top, muted/hatched baselines with a single saturated "Ours" bar as the hero,
and "OOM" annotations where a bar is absent.

Two companion figures share the exact same layout/style:
  * HBM  -- Peak GPU memory (GiB); LOWER is better. This is the memory *saving*:
            AsymLoRA keeps the least resident HBM, so it can push seq length
            further before OOM. A down-arrow % over each Ours bar reports the
            reduction vs the strongest baseline (SuperOffload + Act. Offload).
  * DRAM -- Host RAM (GiB); shows WHERE the memory goes -- the offloading
            trade-off cost that buys the HBM headroom above.

Each model was profiled at its own long-sequence, batch-8 workload, so the
total tokens per step (per-sample seq x 8) is printed under every model name
(the bars are only comparable within a group -- across groups it differs).

Numbers are measured (paper.md Results table, step_H = peak HBM GiB, RAM = host
GiB). Shared fonts/sizes/colors/figure params + the grouped-bar layout helper
all come from constants.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import constants as C  # noqa: E402

# --- Measured data (paper.md Results; all runs batch = 8) -----------------
# hbm = step_H (peak HBM, GiB); dram = RAM (host, GiB). Series keys match
# constants.MEMSAVE_SERIES. A value of None renders an "OOM" marker.
# tokens = total tokens per step (per-sample seq x batch 8), the group label.
MODELS = [
    {"name": "Qwen3-30B-A3B", "tokens": "640k",   # 80k x 8
     "hbm": {"superoffload": 176.9, "superoffload_off": 94.4,  "ours": 73.9},
     "dram": {"superoffload": 360.0, "superoffload_off": 588.5, "ours": 642.2}},
    {"name": "Qwen3-32B", "tokens": "400k",       # 50k x 8
     "hbm": {"superoffload": 180.9, "superoffload_off": 110.7, "ours": 96.4},
     "dram": {"superoffload": 340.4, "superoffload_off": 644.4, "ours": 657.7}},
    {"name": "Qwen2.5-32B", "tokens": "400k",     # 50k x 8
     "hbm": {"superoffload": 171.6, "superoffload_off": 118.4, "ours": 81.2},
     "dram": {"superoffload": 382.7, "superoffload_off": 633.0, "ours": 617.0}},
    {"name": "Qwen2.5-72B", "tokens": "240k",     # 30k x 8
     "hbm": {"superoffload": 125.0, "superoffload_off": 80.8,  "ours": 62.5},
     "dram": {"superoffload": 492.2, "superoffload_off": 662.7, "ours": 733.6}},
    {"name": "Llama-3.3-70B", "tokens": "200k",   # 25k x 8
     "hbm": {"superoffload": 102.2, "superoffload_off": 65.7,  "ours": 52.1},
     "dram": {"superoffload": 489.6, "superoffload_off": 657.6, "ours": 730.4}},
]

# --- Per-metric figure config --------------------------------------------
METRICS = {
    "hbm": {
        "ylabel": "Peak HBM (GiB)",
        "tick_step": 50,
        "filename": "memory_saving_hbm",
        "annotate_saving": True,   # arrow % over each bar vs the bar to its left
    },
    "dram": {
        "ylabel": "Host DRAM (GiB)",
        "tick_step": 200,
        "filename": "memory_saving_dram",
        "annotate_saving": False,  # DRAM is the offloading cost, not a saving
    },
}

_PARAMS = C.FIGURE_PARAMS["main"]
_SERIES = C.MEMSAVE_SERIES
_OURS_KEY = "ours"
_BEST_BASELINE = "superoffload_off"   # strongest baseline -> saving computed vs this


def _values(model, metric):
    return [model[metric][s["key"]] for s in _SERIES]


def build_figure(metric):
    cfg = METRICS[metric]
    C.apply_style()
    pp = _PARAMS
    bar_w = pp["bar_width"]

    bars, centers = C.grouped_layout(
        len(MODELS), len(_SERIES), bar_w, pp["intra_gap"], pp["group_gap"])
    flat = [x for grp in bars for x in grp]
    xlim = C.xlim_from_positions(flat, bar_w, pp["x_margin"])

    fig, ax = plt.subplots(figsize=C.figure_size_for_xlim(xlim, pp["height"]),
                           dpi=pp["dpi"])
    fig.subplots_adjust(left=pp["axes_left"], right=pp["axes_right"],
                        top=pp["axes_top"], bottom=pp["axes_bottom"])

    ymax = max(v for m in MODELS for v in _values(m, metric) if v is not None)

    # grouped bars: one group per model, one bar per system
    for g, model in enumerate(MODELS):
        for s, series in enumerate(_SERIES):
            x = bars[g][s]
            val = model[metric][series["key"]]
            if val is None:
                ax.text(x, ymax * 0.012, "OOM", rotation=90, ha="center",
                        va="bottom", fontsize=C.FONT_SIZE_SEGMENT - 4,
                        color=C.MEMSAVE_OOM_COLOR, fontweight="bold", zorder=4)
                continue
            ax.bar(x, val, width=bar_w, color=series["color"],
                   edgecolor=series["edge"], linewidth=0.8,
                   hatch=series["hatch"], zorder=3)

        # chained arrow %: every bar after the first is annotated vs the bar
        # to its left (Act.Offload vs Recompute, Ours vs Act.Offload)
        if cfg["annotate_saving"]:
            for s in range(1, len(_SERIES)):
                base = model[metric][_SERIES[s - 1]["key"]]
                val = model[metric][_SERIES[s]["key"]]
                if base is None or val is None:
                    continue
                pct = (base - val) / base * 100.0
                arrow = "↓" if pct >= 0 else "↑"
                ax.text(bars[g][s], val + ymax * 0.02, f"{arrow}{abs(pct):.0f}%",
                        ha="center", va="bottom", fontsize=C.FONT_SIZE_SEGMENT,
                        color=C.ACCENT_RED, fontweight="bold", zorder=4)

    ax.set_ylim(0, ymax * 1.22)
    step = cfg["tick_step"]
    tick_top = int(((ymax + step - 1) // step) * step)
    ax.set_yticks(range(0, tick_top + 1, step))
    ax.set_xlim(*xlim)

    # x tick marks hidden; two-line group label (model name + seq len) beneath
    ax.set_xticks(centers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    trans = ax.get_xaxis_transform()
    for center, model in zip(centers, MODELS):
        ax.text(center, pp["model_label_y"], model["name"], transform=trans,
                ha="center", va="top", fontsize=C.FONT_SIZE_MODEL,
                fontweight="bold", clip_on=False)
        ax.text(center, pp["seq_label_y"], model["tokens"],
                transform=trans, ha="center", va="top",
                fontsize=C.FONT_SIZE_SEGMENT, color=C.SUBTLE_TEXT, clip_on=False)

    ax.set_ylabel(cfg["ylabel"])
    # dashed y-grid behind bars + full rectangular box frame (reference style)
    ax.grid(axis="y", color=C.GRID_COLOR, linewidth=0.9, linestyle=(0, (4, 3)),
            alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(C.SPINE_COLOR)
        spine.set_linewidth(1.2)

    handles = [
        Patch(facecolor=s["color"], edgecolor=s["edge"], hatch=s["hatch"],
              label=s["label"])
        for s in _SERIES
    ]
    C.horizontal_legend(ax, handles)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot the AsymLoRA memory-saving figure(s).")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--metric", choices=list(METRICS) + ["both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(METRICS) if args.metric == "both" else [args.metric]
    for metric in metrics:
        fig = build_figure(metric)
        stem = METRICS[metric]["filename"]
        for ext in ("pdf", "png"):   # PDF for the paper, PNG for quick visual QA
            out_path = out_dir / f"{stem}.{ext}"
            fig.savefig(out_path, dpi=_PARAMS["dpi"])
            print(f"wrote {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
