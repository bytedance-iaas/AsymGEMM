#!/usr/bin/env python3
"""LoRA SFT step-time breakdown figure.

Grouped stacked bars: forward / backward / optimizer time per training step for
Qwen3-32B (dense) and Qwen3-MoE-30B, each under ZeRO-3 Offload vs SuperOffload.

Rhetorical point (systems-paper motivation): the optimizer step is a sliver of
LoRA SFT step time, so SuperOffload -- which only accelerates the optimizer /
offload step -- yields ~no end-to-end speedup. It targets the wrong bottleneck.

Shared fonts/sizes/colors/figure params come from constants.py.

NOTE: the timings below are ILLUSTRATIVE PLACEHOLDERS. To plot measured data,
replace the numbers in BARS with forward/backward/optimizer milliseconds from
the per-run step_samples.csv files, e.g.:
  profiling_both*/asym_long_sft_smoke__lora__lf__bf16/<model>__*/<backend>__*/b*/step_samples.csv
(columns: forward_milliseconds, backward_milliseconds, optimizer_milliseconds).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import constants as C  # noqa: E402

PHASES = ["Forward", "Backward", "Optimizer"]

# --- ILLUSTRATIVE PLACEHOLDER TIMINGS (ms / step) -------------------------
# Two model groups, each: ZeRO-3 Offload vs SuperOffload. Story baked into the
# numbers: optimizer ~2-3% of step; SuperOffload shrinks only the optimizer
# slice -> total barely moves (~1.01x).
BARS = [
    {"group": 0, "backend": "Z3 Offload", "Forward": 120.0, "Backward": 300.0, "Optimizer": 14.0},
    {"group": 0, "backend": "SuperOffload", "Forward": 120.0, "Backward": 300.0, "Optimizer": 8.0},
    {"group": 1, "backend": "Z3 Offload", "Forward": 95.0, "Backward": 250.0, "Optimizer": 12.0},
    {"group": 1, "backend": "SuperOffload", "Forward": 95.0, "Backward": 250.0, "Optimizer": 7.0},
]
GROUP_MODELS = ["Qwen3-32B (Dense)", "Qwen3-30B-A3B (MoE)"]
# -------------------------------------------------------------------------

# bar-layout helper now lives in constants.layout()

_PARAMS = C.FIGURE_PARAMS["lora_timing"]


def _total(bar) -> float:
    return sum(bar[p] for p in PHASES)


def build_figure():
    C.apply_style()
    bar_w = _PARAMS["bar_width"]
    xpos, centers = C.layout(_PARAMS["pair_gap"], _PARAMS["group_gap"], len(GROUP_MODELS))
    xlim = C.xlim_from_positions(xpos, bar_w, _PARAMS["x_margin"])
    fig, ax = plt.subplots(figsize=C.figure_size_for_xlim(xlim, _PARAMS["height"]),
                           dpi=_PARAMS["dpi"])
    fig.subplots_adjust(left=_PARAMS["axes_left"], right=_PARAMS["axes_right"],
                        top=_PARAMS["axes_top"], bottom=_PARAMS["axes_bottom"])

    ymax = max(_total(b) for b in BARS)

    for x, bar in zip(xpos, BARS):
        total = _total(bar)
        bottom = 0.0
        for phase in PHASES:
            seg = bar[phase]
            ax.bar(
                x, seg, width=bar_w, bottom=bottom,
                color=C.PHASE_COLORS[phase], edgecolor=C.PHASE_EDGE[phase],
                linewidth=0.6, hatch=C.PHASE_HATCH.get(phase), zorder=3,
            )
            pct = seg / total * 100.0
            if phase == "Optimizer":
                # sliver too thin to hold text -> label just above the bar top
                ax.text(x, total + ymax * 0.015, f"{pct:.1f}%", ha="center",
                        va="bottom", fontsize=C.FONT_SIZE_SEGMENT,
                        color=C.PHASE_EDGE["Optimizer"], fontweight="bold", zorder=4)
            else:
                ax.text(x, bottom + seg / 2.0, f"{pct:.1f}%", ha="center",
                        va="center", fontsize=C.FONT_SIZE_SEGMENT, color="white",
                        fontweight="bold", zorder=4)
            bottom += seg

    ax.set_ylim(0, ymax * 1.24)
    ax.set_xlim(*xlim)

    # per-group speedup, placed ON TOP of the SuperOffload bar (2nd of each pair)
    group_bars = {0: (BARS[0], BARS[1]), 1: (BARS[2], BARS[3])}
    for g in range(len(GROUP_MODELS)):
        z3, so = group_bars[g]
        speedup = _total(z3) / _total(so)
        ax.text(
            xpos[2 * g + 1], _total(so) + ymax * 0.13, f"{speedup:.2f}× Speedup",
            ha="center", va="bottom", fontsize=C.FONT_SIZE_SEGMENT,
            fontweight="bold", color=C.ACCENT_RED,
        )

    # x ticks = backend; model group labels underneath
    ax.set_xticks(xpos)
    ax.set_xticklabels([b["backend"] for b in BARS])
    ax.tick_params(axis="x", length=0, pad=_PARAMS["x_tick_label_pad"])
    trans = ax.get_xaxis_transform()
    for center, model in zip(centers, GROUP_MODELS):
        ax.text(center, _PARAMS["model_label_y"], model, transform=trans,
                ha="center", va="top", fontsize=C.FONT_SIZE_MODEL,
                fontweight="bold", clip_on=False)

    ax.set_ylabel("Time per step (ms)")
    ax.grid(axis="y", color=C.GRID_COLOR, linewidth=0.6, alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # legend: horizontal row above the axes box, colors == segment colors
    handles = [
        Patch(facecolor=C.PHASE_COLORS[p], edgecolor=C.PHASE_EDGE[p],
              hatch=C.PHASE_HATCH.get(p),
              label="Optimizer Update (LoRA)" if p == "Optimizer" else p)
        for p in PHASES
    ]
    C.horizontal_legend(ax, handles)

    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot the LoRA SFT step-time breakdown figure.")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "out"),
                    help="directory to write the figure into")
    ap.add_argument("--filename", default="lora_timing_breakdown.pdf")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out_path = out_dir / args.filename
    fig.savefig(out_path, dpi=_PARAMS["dpi"])
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
