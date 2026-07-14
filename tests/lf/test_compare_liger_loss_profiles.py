from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/lf/archive/compare_liger_loss_profiles.py"
GIB = 1024**3


def _write_run(
    root: Path,
    *,
    backend: str = "asym_cpuadamwds",
    liger_loss: str,
    peak_gib: float,
    lm_head_loss_gib: float,
    step_ms: float,
    forward_ms: float,
    backward_ms: float,
) -> Path:
    run_dir = root / f"{backend}__source__norecomp__ligerloss{liger_loss[-1]}" / "b1_s128"
    run_dir.mkdir(parents=True)
    (run_dir / "source_profile.json").write_text(
        json.dumps(
            {
                "config": {"backend": backend, "liger_loss": liger_loss},
                "trainer": {
                    "timing": {
                        "measured_e2e_step_milliseconds": step_ms,
                    }
                },
                "forward": {"total_milliseconds": forward_ms},
                "backward": {"total_milliseconds": backward_ms},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "memory_breakdown_summary.json").write_text(
        json.dumps(
            {
                "actual_peak_allocated_hbm_bytes": int(peak_gib * GIB),
                "actual_peak_breakdown_rows": [
                    {
                        "component": "lm_head",
                        "kind": "loss_logits",
                        "memory_space": "GPU HBM",
                        "bytes": int(lm_head_loss_gib * GIB),
                    },
                    {
                        "component": "attention",
                        "kind": "activation",
                        "memory_space": "GPU HBM",
                        "bytes": int(1 * GIB),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "step_samples.csv").write_text(
        "\n".join(
            [
                "is_warmup,step_milliseconds,forward_milliseconds,backward_milliseconds",
                f"true,{step_ms * 2},{forward_ms * 2},{backward_ms * 2}",
                f"false,{step_ms},{forward_ms},{backward_ms}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return run_dir


def _run_compare(
    baseline: Path,
    candidate: Path,
    *,
    backend: str = "asym_cpuadamwds",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(SCRIPT),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--backend",
        backend,
        "--baseline-liger-loss",
        "ligerloss0",
        "--candidate-liger-loss",
        "ligerloss1",
        "--min-peak-drop-gib",
        "10",
        "--min-lm-head-loss-drop-gib",
        "20",
        "--max-step-ratio",
        "1.10",
        "--max-forward-ratio",
        "1.15",
        "--max-backward-ratio",
        "1.15",
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def test_compare_liger_loss_profiles_accepts_meaningful_memory_drop(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        liger_loss="ligerloss0",
        peak_gib=100,
        lm_head_loss_gib=32,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )
    candidate = _write_run(
        tmp_path,
        liger_loss="ligerloss1",
        peak_gib=84,
        lm_head_loss_gib=4,
        step_ms=104,
        forward_ms=42,
        backward_ms=61,
    )

    result = _run_compare(baseline, candidate)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["checks"]["peak_drop_gib"] == 16
    assert payload["checks"]["lm_head_loss_drop_gib"] == 28
    assert "memory_breakdown_summary.json" in payload["baseline"]["sources"]["memory_summary"]


def test_compare_liger_loss_profiles_rejects_trivial_memory_drop(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        liger_loss="ligerloss0",
        peak_gib=100,
        lm_head_loss_gib=32,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )
    candidate = _write_run(
        tmp_path,
        liger_loss="ligerloss1",
        peak_gib=96,
        lm_head_loss_gib=30,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )

    result = _run_compare(baseline, candidate)

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "peak HBM drop below threshold" in payload["failures"]
    assert "lm_head/loss HBM drop below threshold" in payload["failures"]


def test_compare_liger_loss_profiles_rejects_latency_regression(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        liger_loss="ligerloss0",
        peak_gib=100,
        lm_head_loss_gib=32,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )
    candidate = _write_run(
        tmp_path,
        liger_loss="ligerloss1",
        peak_gib=80,
        lm_head_loss_gib=4,
        step_ms=130,
        forward_ms=55,
        backward_ms=61,
    )

    result = _run_compare(baseline, candidate)

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "step latency ratio above threshold" in payload["failures"]
    assert "forward latency ratio above threshold" in payload["failures"]


def test_compare_liger_loss_profiles_rejects_missing_lm_head_loss_attribution(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        liger_loss="ligerloss0",
        peak_gib=100,
        lm_head_loss_gib=32,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )
    candidate = _write_run(
        tmp_path,
        liger_loss="ligerloss1",
        peak_gib=80,
        lm_head_loss_gib=4,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )
    summary = json.loads((candidate / "memory_breakdown_summary.json").read_text(encoding="utf-8"))
    summary["actual_peak_breakdown_rows"][0]["component"] = "attention"
    summary["actual_peak_breakdown_rows"][0]["kind"] = "activation"
    (candidate / "memory_breakdown_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = _run_compare(baseline, candidate)

    assert result.returncode == 2
    assert "no GPU HBM rows attributed to lm_head or loss" in result.stdout


def test_compare_liger_loss_profiles_rejects_same_axis(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        liger_loss="ligerloss0",
        peak_gib=100,
        lm_head_loss_gib=32,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(baseline),
            "--backend",
            "asym_cpuadamwds",
            "--baseline-liger-loss",
            "ligerloss0",
            "--candidate-liger-loss",
            "ligerloss0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "same liger_loss" in result.stdout


def test_compare_liger_loss_profiles_rejects_metadata_or_path_mismatch(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        liger_loss="ligerloss0",
        peak_gib=100,
        lm_head_loss_gib=32,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )
    candidate = _write_run(
        tmp_path,
        backend="zero3_offload",
        liger_loss="ligerloss1",
        peak_gib=80,
        lm_head_loss_gib=4,
        step_ms=100,
        forward_ms=40,
        backward_ms=60,
    )

    result = _run_compare(baseline, candidate, backend="asym_cpuadamwds")

    assert result.returncode == 2
    assert "backend mismatch" in result.stdout
