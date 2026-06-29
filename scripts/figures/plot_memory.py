#!/usr/bin/env python3
"""LoRA SFT memory-breakdown figure.

Grouped vertical stacked bars: peak memory split into Model Weights / Optim.
States (LoRA weights+grads) / Activations, per model (Qwen3-32B Dense,
Qwen3-30B-A3B MoE), with PyTorch vs SuperOffload at long sequence length. The
point is that SuperOffload removes most model/optimizer state from GPU memory,
but leaves the activation footprint untouched.

Shared fonts/sizes/colors/figure params + the bar-layout helper come from
constants.py.

NOTE: the exact memory decomposition is hard to measure precisely, so the
numbers below are ILLUSTRATIVE PLACEHOLDERS -- replace them in BARS with
measured GiB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import constants as C  # noqa: E402

COMPONENTS = ["Activations", "Model Weights", "Optim. States"]
STACK_ORDER = COMPONENTS

# --- ILLUSTRATIVE PLACEHOLDER MEMORY (GiB) -------------------------------
# One long-sequence case per model: PyTorch then SuperOffload.
# SuperOffload placeholders keep activations unchanged while shrinking
# model/optimizer resident memory to a small residual.
BARS = [
    {"group": 0, "backend": "PyTorch", "Model Weights": 64.0, "Activations": 96.0,  "Optim. States": 4.0},
    {"group": 0, "backend": "SuperOffload", "Model Weights": 10.0, "Activations": 96.0,  "Optim. States": 2.0},
    {"group": 1, "backend": "PyTorch", "Model Weights": 58.0, "Activations": 112.0, "Optim. States": 5.0},
    {"group": 1, "backend": "SuperOffload", "Model Weights": 10.0, "Activations": 112.0, "Optim. States": 2.0},
]
GROUP_MODELS = ["Qwen3-32B (Dense)", "Qwen3-30B-A3B (MoE)"]
# -------------------------------------------------------------------------

_PARAMS = C.FIGURE_PARAMS["memory"]


def _total(bar) -> float:
    return sum(bar[c] for c in COMPONENTS)


def _layout_memory():
    pp = _PARAMS
    return C.layout(pp["pair_gap"], pp["group_gap"], len(GROUP_MODELS))


def build_figure():
    C.apply_style()
    pp = _PARAMS
    bar_w = pp["bar_width"]
    xpos, centers = _layout_memory()
    xlim = C.xlim_from_positions(xpos, bar_w, pp["x_margin"])
    fig, ax = plt.subplots(figsize=C.figure_size_for_xlim(xlim, pp["height"]),
                           dpi=pp["dpi"])
    fig.subplots_adjust(left=pp["axes_left"], right=pp["axes_right"],
                        top=pp["axes_top"], bottom=pp["axes_bottom"])

    ymax = max(_total(b) for b in BARS)

    # vertical stacked bars: memory on y, stacked bottom -> top
    for x, bar in zip(xpos, BARS):
        total = _total(bar)
        bottom = 0.0
        for comp in STACK_ORDER:
            seg = bar[comp]
            ax.bar(
                x, seg, width=bar_w, bottom=bottom,
                color=C.MEMORY_COLORS[comp], edgecolor=C.MEMORY_EDGE[comp],
                linewidth=0.6, zorder=3,
            )
            pct = seg / total * 100.0
            if "Super" in bar["backend"] and comp in ("Model Weights", "Optim. States"):
                bottom += seg
                continue
            if comp == "Optim. States":
                ax.text(x, total + ymax * 0.015, f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=C.FONT_SIZE_SEGMENT,
                        color="#1F4E79", fontweight="bold", zorder=4)
            elif seg >= ymax * 0.07:
                ax.text(x, bottom + seg / 2.0, f"{pct:.1f}%",
                        ha="center", va="center", fontsize=C.FONT_SIZE_SEGMENT,
                        color="white", fontweight="bold", zorder=4)
            else:
                ax.text(x, bottom + seg * 0.72, f"{pct:.1f}%",
                        ha="center", va="center", fontsize=C.FONT_SIZE_SEGMENT - 6,
                        color="white", fontweight="bold", zorder=4)
            bottom += seg

    # Red boundary lines: activation height is unchanged across each pair.
    for g in range(len(GROUP_MODELS)):
        a, b = 2 * g, 2 * g + 1
        y = BARS[a]["Activations"]
        ax.plot([xpos[a] - bar_w / 2.0, xpos[b] + bar_w / 2.0], [y, y],
                lw=1.7, color="#B3261E", zorder=3.4)

    ax.set_ylim(0, ymax * 1.24)
    tick_top = int(((ymax + 49) // 50) * 50)
    ax.set_yticks(range(0, tick_top + 1, 50))
    ax.set_xlim(*xlim)

    # x ticks = backend; model group labels underneath
    ax.set_xticks(xpos)
    ax.set_xticklabels([b["backend"] for b in BARS])
    ax.tick_params(axis="x", length=0, pad=pp["x_tick_label_pad"])
    trans = ax.get_xaxis_transform()
    for center, model in zip(centers, GROUP_MODELS):
        ax.text(center, pp["model_label_y"], model, transform=trans,
                ha="center", va="top", fontsize=C.FONT_SIZE_MODEL,
                fontweight="bold", clip_on=False)

    ax.set_ylabel("HBM Usage (GiB)")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend_labels = {
        "Model Weights": "Weights",
        "Optim. States": "Optim. States (LoRA)",
    }
    handles = [
        Patch(facecolor=C.MEMORY_COLORS[c], edgecolor=C.MEMORY_EDGE[c],
              label=legend_labels.get(c, c))
        for c in COMPONENTS
    ]
    C.horizontal_legend(ax, handles)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot the LoRA SFT memory-breakdown figure.")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--filename", default="memory_breakdown.pdf")
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
