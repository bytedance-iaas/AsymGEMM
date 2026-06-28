from pathlib import Path

import pytest

from scripts.plotting import plot_lf_interconnect_ctc as plotter


def test_interconnect_timeline_uses_100_points_per_step() -> None:
    assert plotter.POINTS_PER_STEP == 100


def test_interconnect_resample_supports_mean_and_max() -> None:
    series = [(0.1, 10.0), (0.2, 40.0), (0.7, 25.0)]
    bounds = [(0.0, 1.0)]

    mean_points = plotter._resample_per_step(series, bounds, 1, "mean")
    max_points = plotter._resample_per_step(series, bounds, 1, "max")

    assert mean_points == [(pytest.approx(0.5), pytest.approx(25.0))]
    assert max_points == [(pytest.approx(0.5), pytest.approx(40.0))]


def test_interconnect_timeline_plot_writes_avg_and_max(tmp_path: Path) -> None:
    run = plotter.RunRecord(
        run_dir=tmp_path,
        profile_path=tmp_path / "profile.json",
        metadata={"workload": "test-workload"},
        by_step={},
        timelines={
            "avg": {
                "C2C RX": [(0.5, 12.0)],
                "C2C TX": [(0.5, 20.0)],
            },
            "max": {
                "C2C RX": [(0.5, 30.0)],
                "C2C TX": [(0.5, 45.0)],
            },
        },
    )

    plotter._plot_timelines([run], tmp_path)

    assert (tmp_path / "timeline_avg.png").is_file()
    assert (tmp_path / "timeline_max.png").is_file()
    assert not (tmp_path / "timeline.png").exists()
