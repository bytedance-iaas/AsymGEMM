#!/usr/bin/env python3
"""Paper-style C2C/CTC saturation timeline from a per-run metric artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import constants as C  # noqa: E402
from timeline_common import (  # noqa: E402
    affine_to_bounds_window,
    aggregate_max_by_x,
    infer_step_count,
    measured_step_bounds,
    measured_step_phase_spans,
    metric_dir_from_timeseries,
    read_csv_rows,
    resolve_timeseries_path,
    resample_to_measured_steps,
    resample_with_bounds,
)

POINTS_PER_STEP = 50
_PARAMS = C.FIGURE_PARAMS["timeline"]


def build_figure(input_path: Path):
    C.apply_style()
    csv_path = resolve_timeseries_path(input_path, "interconnect")
    metric_dir = metric_dir_from_timeseries(csv_path)
    n_steps = infer_step_count(metric_dir)
    bounds = measured_step_bounds(metric_dir)
    phase_spans = measured_step_phase_spans(metric_dir, bounds)
    rows = read_csv_rows(csv_path)
    series = aggregate_max_by_x(
        rows,
        metric_column="metric_key",
        x_column="timestamp_ms",
    )

    fig, ax = plt.subplots(
        figsize=(_PARAMS["c2c_width"], _PARAMS["height"]),
        dpi=_PARAMS["dpi"],
    )
    fig.subplots_adjust(
        left=_PARAMS["axes_left"],
        right=_PARAMS["axes_right"],
        bottom=_PARAMS["axes_bottom"],
        top=_PARAMS["axes_top"],
    )

    handles = []
    phase_handles = []
    for phase in ("forward", "backward", "optimizer"):
        drawn = False
        for span_phase, x0, x1 in phase_spans:
            if span_phase != phase or x1 <= x0:
                continue
            ax.axvspan(
                x0,
                x1,
                color=C.TIMELINE_PHASE_COLORS[phase],
                alpha=C.TIMELINE_PHASE_ALPHA,
                linewidth=0,
                zorder=0,
            )
            drawn = True
        if drawn:
            phase_handles.append(
                Patch(
                    facecolor=C.TIMELINE_PHASE_COLORS[phase],
                    alpha=C.TIMELINE_PHASE_ALPHA,
                    label=C.TIMELINE_PHASE_LABELS[phase],
                )
            )
    for metric in ("ctc_rx", "ctc_tx"):
        raw_points = series.get(metric, [])
        if bounds:
            points = resample_with_bounds(
                affine_to_bounds_window(raw_points, bounds),
                bounds=bounds,
                points_per_step=POINTS_PER_STEP,
                agg="max",
            )
        else:
            points = resample_to_measured_steps(
                raw_points,
                n_steps=n_steps,
                points_per_step=POINTS_PER_STEP,
                agg="max",
            )
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        line, = ax.plot(
            xs,
            ys,
            color=C.C2C_COLORS[metric],
            linewidth=_PARAMS["line_width"],
            label=C.C2C_LABELS[metric],
            zorder=3,
        )
        handles.append(line)

    for boundary in range(1, n_steps):
        ax.axvline(
            boundary,
            color=_PARAMS["boundary_color"],
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )
    ax.set_xlim(0.0, float(max(n_steps, 1)))
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(list(range(n_steps + 1)))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Step")
    ax.set_ylabel("C2C Saturation (%)")
    ax.grid(axis="y", color=_PARAMS["grid_color"], linewidth=0.6, alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=4)

    if handles:
        C.horizontal_legend(ax, handles + phase_handles, "timeline")
    return fig, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Replot a per-run C2C timeline for paper figures.")
    ap.add_argument("input_path", type=Path, help="run dir, metric dir, timeline PNG, or timeseries CSV")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--filename", default="c2c_timeline.pdf")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, csv_path = build_figure(args.input_path)
    out_path = args.output_dir / args.filename
    fig.savefig(out_path, dpi=_PARAMS["dpi"])
    plt.close(fig)
    print(f"wrote {out_path} from {csv_path}")


if __name__ == "__main__":
    main()
