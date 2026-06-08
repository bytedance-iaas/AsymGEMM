from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asym_gemm.profiling.lf_trace import LFMemoryBreakdownProfiler, LFTraceConfig, build_memory_breakdown_summary  # noqa: E402
from scripts.lf.validate_lf_memory_capacity_schema import validate_breakdown  # noqa: E402
from scripts.lf import run_lf_profiled_train as lf_profiled_train  # noqa: E402


def _load_plotter():
    path = REPO_ROOT / "scripts" / "plotting" / "plot_lf_memory_breakdown.py"
    spec = importlib.util.spec_from_file_location("plot_lf_memory_breakdown", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    phase: str,
    peak_allocated: int,
    peak_reserved: int,
    persistent: dict[str, dict[str, int]] | None = None,
    saved: dict[str, int] | None = None,
    external: dict[str, int] | None = None,
) -> dict:
    persistent = persistent or {}
    saved = saved or {}
    external = external or {}
    known = sum(
        int(value)
        for kinds in persistent.values()
        for kind, value in kinds.items()
        if not str(kind).endswith("_cpu") and not str(kind).endswith("_cpu_pinned")
    ) + sum(int(value) for value in saved.values())
    return {
        "schema_version": 2,
        "step": 1,
        "phase": phase,
        "is_warmup": False,
        "allocated_bytes": peak_allocated,
        "reserved_bytes": peak_reserved,
        "peak_allocated_since_step_begin": peak_allocated,
        "peak_reserved_since_step_begin": peak_reserved,
        "persistent_bytes": persistent,
        "saved_activation_bytes_at_peak": saved,
        "external_memory": external,
        "closure_bytes": {
            "unattributed_allocated_peak": max(0, peak_allocated - known),
            "allocator_reserved_unallocated": max(0, peak_reserved - peak_allocated),
            "external_cuda_or_driver": int(external.get("external_cuda_or_driver_bytes") or 0),
        },
    }


def test_breakdown_selects_allocated_peak_not_reserved_tie() -> None:
    summary = build_memory_breakdown_summary(
        [
            _row(
                phase="step_begin",
                peak_allocated=10,
                peak_reserved=100,
                persistent={"attention": {"weight": 10}},
            ),
            _row(
                phase="after_backward",
                peak_allocated=80,
                peak_reserved=100,
                persistent={
                    "attention": {"weight": 10},
                    "routed_experts": {"grad": 5},
                    "shared_experts": {"optimizer_state": 15},
                },
                saved={"attention": 20},
            ),
        ]
    )

    assert summary["schema_version"] == 2
    assert summary["selected_metric"] == "peak_allocated_hbm_bytes"
    assert summary["selected_phase"] == "after_backward"
    assert summary["peak_allocated_hbm_bytes"] == 80
    assert summary["peak_reserved_hbm_bytes"] == 100
    assert summary["allocated_stack_sum_bytes"] == 80
    assert summary["reserved_stack_sum_bytes"] == 100
    assert summary["saved_activation_hbm_bytes_at_peak"] == 20
    assert summary["unattributed_allocated_peak_bytes"] == 30
    assert summary["allocated_closure_ok"] is True
    assert summary["reserved_closure_ok"] is True

    groups = {row["group"] for row in summary["breakdown_rows"]}
    assert "activations" not in groups
    assert {"weights", "gradients", "optimizer", "saved_activations", "unattributed_allocated_peak"}.issubset(groups)


def test_external_memory_is_diagnostic_not_reserved_closure() -> None:
    summary = build_memory_breakdown_summary(
        [
            _row(
                phase="after_backward",
                peak_allocated=80,
                peak_reserved=100,
                persistent={"attention": {"weight": 10}},
                saved={"routed_experts": 20},
                external={"device_memory_used_bytes": 130, "external_cuda_or_driver_bytes": 30},
            )
        ]
    )

    assert summary["allocated_stack_sum_bytes"] == 80
    assert summary["reserved_stack_sum_bytes"] == 100
    assert summary["external_cuda_or_driver_bytes"] == 30
    assert validate_breakdown(summary) == []


def test_zero_reserved_gap_keeps_allocator_row_for_schema_closure() -> None:
    summary = build_memory_breakdown_summary(
        [
            _row(
                phase="after_backward",
                peak_allocated=80,
                peak_reserved=80,
                persistent={"attention": {"weight": 10}},
                saved={"attention": 20},
            )
        ]
    )

    allocator_rows = [
        row
        for row in summary["breakdown_rows"]
        if row.get("component") == "allocator_reserved_unallocated"
    ]
    assert allocator_rows
    assert allocator_rows[0]["bytes"] == 0
    assert summary["reserved_stack_sum_bytes"] == 80
    assert validate_breakdown(summary) == []


def test_cpu_host_rows_do_not_affect_gpu_closure() -> None:
    summary = build_memory_breakdown_summary(
        [
            _row(
                phase="after_backward",
                peak_allocated=50,
                peak_reserved=70,
                persistent={
                    "attention": {
                        "weight": 10,
                        "weight_cpu": 10_000,
                        "weight_cpu_pinned": 5_000,
                    },
                    "shared_experts": {"optimizer_state": 15},
                },
                saved={"attention": 20},
            )
        ]
    )

    host_rows = [row for row in summary["breakdown_rows"] if str(row.get("memory_space", "")).startswith("CPU")]
    assert host_rows
    assert summary["allocated_stack_sum_bytes"] == 50
    assert summary["reserved_stack_sum_bytes"] == 70
    assert validate_breakdown(summary) == []


def test_warmup_rows_do_not_select_peak_summary() -> None:
    warmup = _row(
        phase="after_backward",
        peak_allocated=200,
        peak_reserved=220,
        persistent={"attention": {"weight": 50}},
    )
    warmup["step"] = 1
    warmup["is_warmup"] = True
    measured = _row(
        phase="after_backward",
        peak_allocated=80,
        peak_reserved=100,
        persistent={"attention": {"weight": 10}},
        saved={"attention": 20},
    )
    measured["step"] = 2

    summary = build_memory_breakdown_summary([warmup, measured])

    assert summary["selected_step"] == 2
    assert summary["peak_allocated_hbm_bytes"] == 80
    assert validate_breakdown(summary) == []


def test_validator_rejects_old_activation_delta_schema() -> None:
    summary = {
        "enabled": True,
        "schema_version": 1,
        "selected_metric": "peak_reserved_hbm_bytes",
        "peak_allocated_hbm_bytes": 10,
        "peak_reserved_hbm_bytes": 12,
        "reserved_unallocated_bytes": 2,
        "allocated_stack_sum_bytes": 10,
        "reserved_stack_sum_bytes": 12,
        "saved_activation_hbm_bytes_at_peak": 0,
        "unattributed_allocated_peak_bytes": 0,
        "allocated_bytes": 10,
        "reserved_bytes": 12,
        "allocated_closure_ok": True,
        "reserved_closure_ok": True,
        "allocated_closure_error_bytes": 0,
        "reserved_closure_error_bytes": 0,
        "activation_bytes": {"attention": 999},
        "breakdown_rows": [
            {"memory_space": "GPU HBM", "group": "activations", "component": "attention", "bytes": 10},
            {
                "memory_space": "GPU reserved",
                "group": "allocator",
                "component": "allocator_reserved_unallocated",
                "bytes": 2,
            },
        ],
    }

    errors = validate_breakdown(summary)
    assert any("schema_version must be 2" in error for error in errors)
    assert any("old activations group is not allowed" in error for error in errors)
    assert any("legacy key" in error for error in errors)


def test_plotter_step_series_uses_component_group_segments(tmp_path: Path) -> None:
    plotter = _load_plotter()
    rows = [
        _row(
            phase="step_begin",
            peak_allocated=10,
            peak_reserved=100,
            persistent={"attention": {"weight": 10}},
        ),
        _row(
            phase="after_backward",
            peak_allocated=80,
            peak_reserved=100,
            persistent={
                "attention": {"weight": 10},
                "routed_experts": {"grad": 5},
                "shared_experts": {"optimizer_state": 15},
            },
            saved={"attention": 20},
        ),
    ]
    summary = build_memory_breakdown_summary(rows)
    jsonl_path = tmp_path / "memory_breakdown.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    record = plotter.RunRecord(
        run_dir=tmp_path,
        summary_path=tmp_path / "memory_breakdown_summary.json",
        jsonl_path=jsonl_path,
        summary=summary,
        metadata={"backend": "asym", "profiler": "source", "seq_len": "4096"},
    )

    steps, series, peak_allocated = plotter._step_series(record)

    assert steps == [1]
    assert peak_allocated == [80]
    assert "attention:weights" in series
    assert "attention:saved_activations" in series
    assert "routed_experts:gradients" in series
    assert "shared_experts:optimizer" in series
    assert "unattributed_allocated_peak" in series
    assert "allocator_reserved_unallocated" in series
    assert all("activations" != key for key in series)


def test_plotter_reports_legacy_schema_when_no_v2_runs_match(tmp_path: Path) -> None:
    plotter = _load_plotter()
    run_dir = tmp_path / "config" / "zero3_offload__source__recomp__polnone__routerhf" / "b1_s4096"
    run_dir.mkdir(parents=True)
    (run_dir / "memory_breakdown_summary.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "schema_version": 1,
                "breakdown_rows": [
                    {"memory_space": "GPU HBM", "group": "temp_workspace", "component": "framework_temp_workspace", "bytes": 1}
                ],
            }
        ),
        encoding="utf-8",
    )

    args = type(
        "Args",
        (),
        {
            "input_root": [tmp_path],
            "run_dir": [],
            "include_non_source": False,
            "workload": [],
            "backend": [],
            "profiler": [],
            "router_mode": [],
            "seq_lens": [],
            "expert_recompute_policies": [],
        },
    )()

    assert plotter._load_runs(args) == []
    message = plotter._no_runs_message(args)
    assert "legacy/non-v2" in message
    assert "schema_version 2" in message


def test_source_recorder_can_preserve_step_peak_counter(monkeypatch) -> None:
    reset_calls = []

    @contextmanager
    def noop_range(_name: str):
        yield

    monkeypatch.setattr(lf_profiled_train, "prof_range", noop_range)
    monkeypatch.setattr(lf_profiled_train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lf_profiled_train.torch.cuda, "memory_allocated", lambda: 10)
    monkeypatch.setattr(lf_profiled_train.torch.cuda, "memory_reserved", lambda: 20)
    monkeypatch.setattr(lf_profiled_train.torch.cuda, "max_memory_allocated", lambda: 30)
    monkeypatch.setattr(lf_profiled_train.torch.cuda, "max_memory_reserved", lambda: 40)
    monkeypatch.setattr(lf_profiled_train.torch.cuda, "reset_peak_memory_stats", lambda: reset_calls.append("reset"))

    preserving = lf_profiled_train.LFProfileRecorder(config={}, reset_stage_peak_stats=False)
    with preserving.stage("step.forward"):
        pass
    assert reset_calls == []

    stage_local = lf_profiled_train.LFProfileRecorder(config={}, reset_stage_peak_stats=True)
    with stage_local.stage("step.forward"):
        pass
    assert reset_calls == ["reset"]


def test_memory_breakdown_restore_clears_hook_marker() -> None:
    class DummyModel:
        pass

    model = DummyModel()
    setattr(model, "_asym_lf_memory_breakdown_hooks_installed", True)
    profiler = LFMemoryBreakdownProfiler(LFTraceConfig(memory_breakdown=True))
    profiler._model = model

    profiler.restore()

    assert getattr(model, "_asym_lf_memory_breakdown_hooks_installed") is False
