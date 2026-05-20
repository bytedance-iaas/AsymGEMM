from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROFILE_CLI = ROOT / "scripts" / "profile_asymgemm_sft.py"

WORKLOAD_REPORTS = {
    "mlp": "m4_5_mlp_profile.json",
    "dense_llm": "m4_5_dense_llm_profile.json",
    "tiny_moe": "m4_5_tiny_moe_profile.json",
}
WORKLOAD_ALIASES = {
    "mlp": ("mlp",),
    "dense_llm": ("dense_llm", "dense llm", "dense"),
    "tiny_moe": ("tiny_moe", "tiny moe", "moe"),
}
SUMMARY_JSON = "m4_5_profile_summary.json"
SUMMARY_MD = "m4_5_profile_summary.md"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict), f"{path} must contain a JSON object"
    return data


def _available_reports(base: Path) -> bool:
    expected = [*WORKLOAD_REPORTS.values(), SUMMARY_JSON, SUMMARY_MD]
    return all((base / name).exists() for name in expected)


def _help_text() -> str:
    result = subprocess.run(
        [sys.executable, str(PROFILE_CLI), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=30,
    )
    return result.stdout


def _flag(help_text: str, candidates: Iterable[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in help_text), None)


def _ensure_reports(tmp_path: Path) -> Path:
    if _available_reports(REPORTS):
        return REPORTS

    assert PROFILE_CLI.exists(), (
        "M4.5 profile reports are missing and scripts/profile_asymgemm_sft.py is not available "
        "to generate lightweight reports"
    )

    output_dir = tmp_path / "m4_5_profile_reports"
    help_text = _help_text()
    command = [sys.executable, str(PROFILE_CLI)]

    output_flag = _flag(help_text, ("--output-dir", "--reports-dir", "--report-dir"))
    assert output_flag is not None, "profiling CLI must expose an output directory flag"
    command += [output_flag, str(output_dir)]

    warmup_flag = _flag(help_text, ("--warmup", "--warmups", "--warmup-steps", "--num-warmup"))
    if warmup_flag is not None:
        command += [warmup_flag, "0"]

    measured_flag = _flag(help_text, ("--measured", "--measured-steps", "--measurements", "--iters", "--steps"))
    if measured_flag is not None:
        command += [measured_flag, "1"]

    lightweight_flag = _flag(help_text, ("--lightweight", "--quick", "--smoke"))
    if lightweight_flag is not None:
        command.append(lightweight_flag)

    backend_flag = _flag(help_text, ("--backend", "--profile-backend"))
    if backend_flag is not None and "torch_only" in help_text:
        command += [backend_flag, "torch_only"]

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert result.returncode == 0, f"profiling CLI failed:\n{' '.join(command)}\n{result.stdout}"
    assert _available_reports(output_dir), f"profiling CLI did not generate the full M4.5 report set:\n{result.stdout}"
    return output_dir


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _key_text(mapping: Mapping[str, Any]) -> str:
    return " ".join(str(key).lower() for key in mapping)


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str).lower()


def _contains_key(report: Mapping[str, Any], *needles: str) -> bool:
    return any(all(needle in _key_text(item) for needle in needles) for item in _walk_mappings(report))


def _numeric(mapping: Mapping[str, Any], keys: Iterable[str]) -> list[float]:
    values: list[float] = []
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _percent_values(mapping: Mapping[str, Any]) -> list[float]:
    return [
        float(value)
        for key, value in mapping.items()
        if "percent" in str(key).lower() and isinstance(value, int | float) and not isinstance(value, bool)
    ]


def _has_total_and_percent(section: Mapping[str, Any], total_keys: tuple[str, ...]) -> bool:
    if _numeric(section, total_keys) and _percent_values(section):
        return True
    for child in section.values():
        if isinstance(child, Mapping) and _has_total_and_percent(child, total_keys):
            return True
        if isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping) and _has_total_and_percent(item, total_keys):
                    return True
    return False


def _find_section(report: Mapping[str, Any], *needles: str) -> Mapping[str, Any]:
    matches = [item for item in _walk_mappings(report) if all(needle in _key_text(item) for needle in needles)]
    assert matches, f"missing section with keys containing: {needles}"
    return matches[0]


def _assert_latency_schema(report: Mapping[str, Any]) -> None:
    latency = _find_section(report, "latency")
    assert _has_total_and_percent(
        latency,
        ("total", "total_ms", "total_seconds", "step_ms", "step_seconds", "wall_ms", "cuda_ms"),
    ), "latency section must include totals and percent-of-total breakdown"
    assert _contains_key(latency, "forward"), "latency report must expose forward timing"
    assert _contains_key(latency, "backward"), "latency report must expose backward timing"
    assert _contains_key(latency, "optimizer"), "latency report must expose optimizer timing"


def _assert_memory_schema(report: Mapping[str, Any]) -> None:
    cpu = _find_section(report, "cpu", "memory")
    assert _has_total_and_percent(
        cpu,
        ("total", "total_bytes", "rss_bytes", "peak_rss_bytes", "pinned_bytes", "cpu_bytes"),
    ), "CPU memory section must include totals and percent breakdown"
    assert _contains_key(cpu, "pinned"), "CPU memory section must expose pinned/page-locked bytes"
    assert _contains_key(cpu, "rss"), "CPU memory section must expose RSS"

    gpu = _find_section(report, "gpu", "memory")
    assert _has_total_and_percent(
        gpu,
        ("total", "total_bytes", "peak_hbm_bytes", "peak_allocated_bytes", "allocated_bytes", "gpu_bytes"),
    ), "GPU memory section must include totals and percent breakdown"
    assert _contains_key(gpu, "peak"), "GPU memory section must expose peak HBM/allocation"


def _assert_timing_stats(report: Mapping[str, Any]) -> None:
    timing = _find_section(report, "warmup", "measured")
    assert _contains_key(timing, "mean"), "timing stats must include mean"
    assert _contains_key(timing, "median") or _contains_key(timing, "p50"), "timing stats must include median/p50"
    assert _contains_key(timing, "p95"), "timing stats must include p95"
    assert _contains_key(timing, "std") or _contains_key(timing, "cv"), "timing stats must include variance/stability"


def _assert_backend_and_hardware(report: Mapping[str, Any]) -> None:
    assert _contains_key(report, "hardware") or _contains_key(report, "environment"), "missing hardware metadata"
    assert _contains_key(report, "backend"), "missing backend metadata"
    assert _contains_key(report, "fallback"), "missing fallback stats"


def _assert_unattributed_or_estimated_visible(report: Mapping[str, Any]) -> None:
    visible = [
        item
        for item in _walk_mappings(report)
        if "unattributed" in _json_text(item) or "estimated" in _json_text(item) or item.get("estimated") is True
    ]
    assert visible, "reports must keep estimated/unattributed accounting visible when attribution is incomplete"


def _assert_workload_report(path: Path, workload: str) -> None:
    report = _load_json(path)
    assert _contains_key(report, "workload") or workload in path.name
    _assert_latency_schema(report)
    _assert_memory_schema(report)
    _assert_backend_and_hardware(report)
    _assert_timing_stats(report)
    _assert_unattributed_or_estimated_visible(report)


def test_m4_5_per_workload_profile_reports_have_required_breakdowns(tmp_path: Path) -> None:
    report_dir = _ensure_reports(tmp_path)
    for workload, name in WORKLOAD_REPORTS.items():
        _assert_workload_report(report_dir / name, workload)


def test_m4_5_cross_workload_summary_json_and_markdown(tmp_path: Path) -> None:
    report_dir = _ensure_reports(tmp_path)
    summary = _load_json(report_dir / SUMMARY_JSON)
    summary_text = _json_text(summary)
    for workload, aliases in WORKLOAD_ALIASES.items():
        assert any(alias in summary_text for alias in aliases), f"summary JSON missing {workload}"

    markdown = (report_dir / SUMMARY_MD).read_text(encoding="utf-8").lower()
    for phrase in ("mlp", "dense", "moe", "latency", "gpu", "cpu", "hbm", "pinned"):
        assert phrase in markdown, f"summary markdown missing {phrase}"
