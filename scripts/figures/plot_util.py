#!/usr/bin/env python3
"""Paper-style GPU/CPU utilization timeline from a per-run metric artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import constants as C  # noqa: E402
from timeline_common import (  # noqa: E402
    affine_to_bounds_window,
    infer_step_count,
    measured_step_bounds,
    measured_step_phase_spans,
    metric_dir_from_timeseries,
    opt_float,
    read_csv_rows,
    resolve_timeseries_path,
    resample_to_measured_steps,
    resample_with_bounds,
)

POINTS_PER_STEP = 100
_PARAMS = C.FIGURE_PARAMS["timeline"]


def _series_by_device(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float]]]:
    series = {"gpu": [], "cpu": []}
    for row in rows:
        device = str(row.get("device") or "")
        if device not in series:
            continue
        if device == "gpu":
            x = opt_float(row.get("timestamp_ms"))
        else:
            x = opt_float(row.get("elapsed_seconds"))
        y = opt_float(row.get("value_percent"))
        if x is None or y is None:
            continue
        series[device].append((x, min(100.0, max(0.0, y))))
    for device in series:
        series[device].sort(key=lambda item: item[0])
    return series


def build_figure(input_path: Path):
    C.apply_style()
    csv_path = resolve_timeseries_path(input_path, "utilization")
    metric_dir = metric_dir_from_timeseries(csv_path)
    n_steps = infer_step_count(metric_dir)
    bounds = measured_step_bounds(metric_dir)
    phase_spans = measured_step_phase_spans(metric_dir, bounds)
    raw = _series_by_device(read_csv_rows(csv_path))
    if bounds:
        points = {
            "gpu": resample_with_bounds(
                affine_to_bounds_window(raw["gpu"], bounds),
                bounds=bounds,
                points_per_step=POINTS_PER_STEP,
                agg="mean",
            ),
            "cpu": resample_with_bounds(
                raw["cpu"],
                bounds=bounds,
                points_per_step=POINTS_PER_STEP,
                agg="mean",
            ),
        }
    else:
        points = {
            device: resample_to_measured_steps(
                values,
                n_steps=n_steps,
                points_per_step=POINTS_PER_STEP,
                agg="mean",
            )
            for device, values in raw.items()
        }

    fig, ax = plt.subplots(
        figsize=(_PARAMS["width"], _PARAMS["height"]),
        dpi=_PARAMS["dpi"],
    )
    fig.subplots_adjust(
        left=_PARAMS["axes_left"],
        right=_PARAMS["axes_right"],
        bottom=_PARAMS["axes_bottom"],
        top=_PARAMS["axes_top"],
    )

    legend_handles: list[Line2D] = []
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
    for device in ("gpu", "cpu"):
        xs = [point[0] for point in points[device]]
        ys = [point[1] for point in points[device]]
        line, = ax.plot(
            xs,
            ys,
            color=C.UTIL_COLORS[device],
            linewidth=_PARAMS["line_width"],
            label=C.UTIL_LABELS[device],
            zorder=3,
        )
        legend_handles.append(line)

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
    ax.set_ylabel("Utilization (%)")
    ax.grid(axis="y", color=_PARAMS["grid_color"], linewidth=0.6, alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=4)
    C.horizontal_legend(ax, legend_handles + phase_handles, "timeline")
    return fig, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Replot a per-run GPU/CPU utilization timeline for paper figures.")
    ap.add_argument("input_path", type=Path, help="run dir, metric dir, utilization PNG, or timeseries CSV")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--filename", default="utilization_timeline.pdf")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, csv_path = build_figure(args.input_path)
    out_path = args.output_dir / args.filename
    fig.savefig(out_path, dpi=_PARAMS["dpi"])
    plt.close(fig)
    print(f"wrote {out_path} from {csv_path}")


if __name__ == "__main__":
    main()
