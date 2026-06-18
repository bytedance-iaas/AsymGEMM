from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asym_gemm.profiling.lf_trace import (  # noqa: E402
    LFMemoryBreakdownProfiler,
    LFTraceConfig,
    _component_from_range_name,
    _component_from_module_name,
    _component_from_param_name,
    build_memory_breakdown_summary,
)
from asym_gemm.integrations.lf import classify_lf_component, component_is_selected, parse_lf_offload_modules  # noqa: E402
from asym_gemm.training.frozen_linear import AsymExecutionStats  # noqa: E402
from scripts.lf.validate_lf_memory_capacity_schema import validate_breakdown  # noqa: E402
from scripts.lf import run_lf_profiled_train as lf_profiled_train  # noqa: E402
import torch  # noqa: E402


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
    live: dict[str, int] | None = None,
    peak_growth: dict[str, int] | None = None,
    external: dict[str, int] | None = None,
) -> dict:
    persistent = persistent or {}
    saved = saved or {}
    live = live or {}
    peak_growth = peak_growth or {}
    external = external or {}
    known = sum(
        int(value)
        for kinds in persistent.values()
        for kind, value in kinds.items()
        if not str(kind).endswith("_cpu") and not str(kind).endswith("_cpu_pinned")
    ) + sum(int(value) for value in saved.values()) + sum(int(value) for value in live.values())
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
        "live_activation_bytes_at_peak": live,
        "peak_growth_bytes_at_peak": peak_growth,
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
    assert summary["live_activation_hbm_bytes_at_peak"] == 0
    assert summary["activation_hbm_bytes_at_peak"] == 20
    assert summary["temporary_workspace_hbm_bytes_at_peak"] == 0
    assert summary["unattributed_allocated_peak_bytes"] == 30
    assert summary["allocated_closure_ok"] is True
    assert summary["reserved_closure_ok"] is True

    groups = {row["group"] for row in summary["breakdown_rows"]}
    assert "activations" not in groups
    assert {"weights", "gradients", "optimizer", "saved_activations", "unattributed_allocated_peak"}.issubset(groups)


def test_qwen_moe_component_classification_separates_router_dense_and_experts() -> None:
    assert _component_from_param_name("model.layers.0.self_attn.q_proj.weight") == "attention"
    assert _component_from_param_name("model.layers.0.mlp.gate.weight") == "router"
    assert _component_from_param_name("model.layers.0.mlp.gate_proj.weight") == "mlp_dense"
    assert _component_from_param_name("model.layers.0.mlp.experts.3.down_proj.weight") == "routed_experts"
    assert _component_from_param_name("model.layers.0.mlp.shared_expert.down_proj.weight") == "shared_experts"
    assert _component_from_param_name("model.layers.0.mlp.experts.3.lora_A.default.weight") == "routed_experts"

    assert _component_from_module_name("model.layers.0.self_attn") == "attention"
    assert _component_from_module_name("model.layers.0.mlp.gate") == "router"
    assert _component_from_module_name("model.layers.0.mlp") == "mlp_dense"
    assert _component_from_module_name("model.layers.0.mlp.experts.3") == "routed_experts"
    assert _component_from_module_name("model.layers.0.mlp.shared_expert") == "shared_experts"

    assert _component_from_range_name("lf.forward_loss") == "loss"
    assert _component_from_range_name("forward.mlp.expert_policy.scatter_combine") == "routed_experts"
    assert _component_from_range_name("backward.layers.0.self_attn.q_proj") == "attention"


def test_qwen35_linear_attention_component_is_profile_only_before_offload_stage() -> None:
    assert _component_from_param_name("model.layers.0.linear_attn.in_proj_qkv.weight") == "linear_attention"
    assert _component_from_param_name("model.layers.0.linear_attn.in_proj_z.lora_A.default.weight") == "linear_attention"
    assert _component_from_param_name("model.layers.0.linear_attn.out_proj.weight") == "linear_attention"
    assert _component_from_module_name("model.layers.0.linear_attn") == "linear_attention"
    assert _component_from_module_name("model.layers.0.linear_attn.in_proj_b") == "linear_attention"
    assert _component_from_range_name("forward.layers.0.linear_attn.chunk_gated_delta_rule") == "linear_attention"
    assert _component_from_range_name("backward.Qwen3_5MoeGatedDeltaNet") == "linear_attention"
    assert classify_lf_component("model.layers.0.linear_attn.in_proj_a.weight") == "linear_attention"

    selection = parse_lf_offload_modules("all")
    assert not component_is_selected("linear_attention", "in_proj_qkv", selection)


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


def test_legacy_unknown_saved_activation_is_reported_as_other() -> None:
    plotter = _load_plotter()
    row = _row(
        phase="after_forward",
        peak_allocated=80,
        peak_reserved=100,
        persistent={"attention": {"weight": 10}},
        saved={"unknown_saved_activation": 20},
    )

    summary = build_memory_breakdown_summary([row])
    summary_components = {item["component"] for item in summary["breakdown_rows"]}
    assert "other_saved_activations" in summary_components
    assert "unknown_saved_activation" not in summary_components

    plot_rows, _peak_allocated, _peak_reserved = plotter._flatten_row(row)
    plot_components = {item["component"] for item in plot_rows}
    assert "other_saved_activations" in plot_components
    assert "unknown_saved_activation" not in plot_components


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


def test_live_activations_and_peak_workspace_reduce_unattributed_residual() -> None:
    summary = build_memory_breakdown_summary(
        [
            _row(
                phase="after_forward",
                peak_allocated=100,
                peak_reserved=120,
                persistent={"attention": {"weight": 10}},
                saved={"attention": 20},
                live={"attention": 15},
                peak_growth={"attention": 65},
            )
        ]
    )

    assert summary["saved_activation_hbm_bytes_at_peak"] == 20
    assert summary["live_activation_hbm_bytes_at_peak"] == 15
    assert summary["activation_hbm_bytes_at_peak"] == 35
    assert summary["temporary_workspace_hbm_bytes_at_peak"] == 55
    assert summary["unattributed_allocated_peak_bytes"] == 0
    assert summary["allocated_stack_sum_bytes"] == 100
    assert validate_breakdown(summary) == []

    rows = {
        (row["group"], row["kind"], row["component"]): row["bytes"]
        for row in summary["breakdown_rows"]
        if row.get("memory_space") == "GPU HBM"
    }
    assert rows[("saved_activations", "saved_activation", "attention")] == 20
    assert rows[("saved_activations", "live_activation", "attention")] == 15
    assert rows[("temporary_workspace", "inferred_peak_workspace", "attention")] == 55


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


def test_lf_profile_report_includes_global_asym_execution_stats(monkeypatch) -> None:
    model = torch.nn.Module()
    stats = AsymExecutionStats(asym_forward_calls=2, asym_dx_calls=1, torch_forward_calls=3)
    setattr(model, "_asym_execution_stats", stats)
    monkeypatch.setattr(lf_profiled_train, "_LAST_LF_MODEL", model)

    recorder = lf_profiled_train.LFProfileRecorder(config={})
    report = recorder.report(None)

    asym_stats = report["asym_execution_stats"]
    assert asym_stats["available"] is True
    assert asym_stats["source"] == "model"
    assert asym_stats["asym_calls"] == 3
    assert asym_stats["torch_calls"] == 3
    assert asym_stats["forward_calls_total"] == 5
    assert asym_stats["backward_calls_total"] == 1


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


def test_summary_selects_same_step_saved_activation_row_for_stale_residual_peak() -> None:
    after_forward = _row(
        phase="after_forward",
        peak_allocated=80,
        peak_reserved=100,
        persistent={"attention": {"weight": 10}},
        saved={"attention": 20},
    )
    after_backward = _row(
        phase="after_backward",
        peak_allocated=100,
        peak_reserved=120,
        persistent={"attention": {"weight": 10}},
        saved={},
    )

    summary = build_memory_breakdown_summary([after_forward, after_backward])

    assert summary["selected_phase"] == "after_forward"
    assert summary["peak_allocated_hbm_bytes"] == 80
    assert summary["saved_activation_hbm_bytes_at_peak"] == 20
    assert summary["unattributed_allocated_peak_bytes"] == 50
    assert summary["actual_peak_phase"] == "after_backward"
    assert summary["actual_peak_allocated_hbm_bytes"] == 100
    assert summary["actual_peak_reserved_hbm_bytes"] == 120
    assert summary["actual_peak_allocated_closure_error_bytes"] == 0
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


def test_plotter_phase_rows_expose_optimizer_weights_gradients_and_saved_activations(tmp_path: Path) -> None:
    plotter = _load_plotter()
    rows = [
        _row(
            phase="after_forward",
            peak_allocated=180,
            peak_reserved=220,
            persistent={
                "attention": {"weight": 10},
                "router": {"weight": 5},
                "mlp_dense": {"weight": 11},
                "routed_experts": {"weight": 13},
                "shared_experts": {"weight": 17},
            },
            saved={
                "attention": 19,
                "router": 3,
                "mlp_dense": 23,
                "routed_experts": 29,
                "shared_experts": 31,
            },
        ),
        _row(
            phase="after_backward",
            peak_allocated=210,
            peak_reserved=240,
            persistent={
                "attention": {"weight": 10, "grad": 10},
                "router": {"weight": 5, "grad": 5},
                "mlp_dense": {"weight": 11, "grad": 11},
                "routed_experts": {"weight": 13, "grad": 13},
                "shared_experts": {"weight": 17, "grad": 17},
            },
        ),
        _row(
            phase="after_optimizer_step",
            peak_allocated=260,
            peak_reserved=300,
            persistent={
                "attention": {"weight": 10, "optimizer_state": 20},
                "router": {"weight": 5, "optimizer_state": 10},
                "mlp_dense": {"weight": 11, "optimizer_state": 22},
                "routed_experts": {"weight": 13, "optimizer_state": 26},
                "shared_experts": {"weight": 17, "optimizer_state": 34},
            },
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
        metadata={"backend": "source_smoke", "profiler": "source", "seq_len": "128"},
    )

    phase_rows = plotter._phase_csv_rows(record)
    pairs = {(row["component"], row["group"]) for row in phase_rows if row["memory_space"] == "GPU HBM"}

    for component in {"attention", "router", "mlp_dense", "routed_experts", "shared_experts"}:
        assert (component, "weights") in pairs
        assert (component, "saved_activations") in pairs
        assert (component, "gradients") in pairs
        assert (component, "optimizer") in pairs

    labels, series, peaks = plotter._phase_plot_data(record)
    assert labels == ["after_forward", "after_backward", "after_optimizer_step"]
    assert "attention:saved_activations" in series
    assert "routed_experts:optimizer" in series
    assert len(peaks) == 3

    actual_rows = plotter._actual_peak_csv_rows(record)
    assert actual_rows
    assert {row["component"] for row in actual_rows if row["memory_space"] == "GPU HBM"} >= {
        "attention",
        "router",
        "mlp_dense",
        "routed_experts",
        "shared_experts",
    }
    assert all("actual_peak_allocated_hbm_bytes" in row for row in actual_rows)


def test_plotter_phase_rows_include_live_activations_and_temporary_workspace(tmp_path: Path) -> None:
    plotter = _load_plotter()
    rows = [
        _row(
            phase="after_forward",
            peak_allocated=100,
            peak_reserved=120,
            persistent={"attention": {"weight": 10}},
            saved={"attention": 20},
            live={"attention": 15},
            peak_growth={"attention": 65},
        )
    ]
    summary = build_memory_breakdown_summary(rows)
    jsonl_path = tmp_path / "memory_breakdown.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    record = plotter.RunRecord(
        run_dir=tmp_path,
        summary_path=tmp_path / "memory_breakdown_summary.json",
        jsonl_path=jsonl_path,
        summary=summary,
        metadata={"backend": "source_smoke", "profiler": "source", "seq_len": "128"},
    )

    phase_rows = plotter._phase_csv_rows(record)
    pairs = {
        (row["component"], row["group"], row["kind"]): row["bytes"]
        for row in phase_rows
        if row["memory_space"] == "GPU HBM"
    }

    assert pairs[("attention", "saved_activations", "saved_activation")] == 20
    assert pairs[("attention", "saved_activations", "live_activation")] == 15
    assert pairs[("attention", "temporary_workspace", "inferred_peak_workspace")] == 55
    assert ("unattributed_allocated_peak", "unattributed_allocated_peak", "allocated_residual") not in pairs
    assert plotter._segment_label("attention:saved_activations") == "Attention activations"
    assert plotter._segment_label("attention:temporary_workspace") == "Attention temporary workspace"


def test_plotter_uses_distinct_colors_and_tight_small_peak_ylim() -> None:
    plotter = _load_plotter()
    required_keys = [
        f"{component}:{group}"
        for component in ["attention", "router", "mlp_dense", "routed_experts", "shared_experts"]
        for group in ["weights", "gradients", "optimizer", "saved_activations"]
    ]
    colors = [plotter._segment_color(key) for key in required_keys]

    assert len(set(colors)) == len(required_keys)
    assert plotter._segment_color("allocator_reserved_unallocated") == "#4f46e5"
    assert plotter._segment_label("other_saved_activations:saved_activations") == "Other activations"
    assert plotter._nice_y_limit_gib(29_360_128) == 0.05


def test_plotter_disambiguates_duplicate_combined_labels() -> None:
    plotter = _load_plotter()

    def record(config: str) -> object:
        run_dir = Path("/tmp") / config / "zero3_offload__source__recomp__polnone__routerhf" / "b4_s8192"
        return plotter.RunRecord(
            run_dir=run_dir,
            summary_path=run_dir / "memory_breakdown_summary.json",
            jsonl_path=run_dir / "memory_breakdown.jsonl",
            summary={"enabled": True, "schema_version": 2},
            metadata={
                "workload": "qwen3-30b-a3b",
                "backend": "zero3_offload",
                "profiler": "source",
                "recompute": "recomp",
                "expert_policy": "none",
                "router_mode": "hf",
                "seq_len": "8192",
                "config": config,
            },
        )

    labels = plotter._run_plot_labels(
        [
            record("qwen3-30b-a3b__gpus1__b4_s8192_w5_s10_r64_a16_drop010"),
            record("qwen3-30b-a3b__gpus1__b4_s8192_w5_s1_r64_a16_drop010"),
        ]
    )

    assert "w5_s10" in labels[0]
    assert "w5_s1" in labels[1]


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
                "expact": [],
                "attnact": [],
                "seq_lens": [],
                "expert_recompute_policies": [],
        },
    )()

    assert plotter._load_runs(args) == []
    message = plotter._no_runs_message(args)
    assert "legacy/non-v2" in message
    assert "schema_version 2" in message


def test_plotter_repairs_stale_summary_from_jsonl(tmp_path: Path) -> None:
    plotter = _load_plotter()
    run_dir = tmp_path / "config" / "zero3_offload__source__recomp__polnone__routerhf" / "b1_s4096"
    run_dir.mkdir(parents=True)
    after_forward = _row(
        phase="after_forward",
        peak_allocated=80,
        peak_reserved=100,
        persistent={"attention": {"weight": 10}},
        saved={"attention": 20},
    )
    after_backward = _row(
        phase="after_backward",
        peak_allocated=100,
        peak_reserved=120,
        persistent={"attention": {"weight": 10}},
        saved={},
    )
    (run_dir / "memory_breakdown.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [after_forward, after_backward]) + "\n",
        encoding="utf-8",
    )
    bad_summary = build_memory_breakdown_summary([after_backward])
    (run_dir / "memory_breakdown_summary.json").write_text(json.dumps(bad_summary), encoding="utf-8")

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
                "expact": [],
                "attnact": [],
                "seq_lens": [],
                "expert_recompute_policies": [],
        },
    )()

    runs = plotter._load_runs(args)

    assert len(runs) == 1
    assert runs[0].summary["saved_activation_hbm_bytes_at_peak"] == 20
    assert runs[0].summary["unattributed_allocated_peak_bytes"] == 50


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


def test_saved_activation_peak_snapshot_is_not_cleared_by_stale_capture() -> None:
    profiler = LFMemoryBreakdownProfiler(LFTraceConfig(memory_breakdown=True))
    profiler._active = True
    key = ("cuda:0", 1234, 4096, "torch.bfloat16")
    profiler._saved_refcounts[key] = 1
    profiler._saved_components[key] = "attention"
    profiler._saved_bytes[key] = 4096

    profiler._capture_saved_activation_peak(100)

    assert profiler._saved_activation_peak_allocated == 100
    assert profiler._saved_activation_bytes_at_peak == {"attention": 4096}

    profiler._saved_refcounts.clear()
    profiler._saved_components.clear()
    profiler._saved_bytes.clear()

    profiler._capture_saved_activation_peak(100)
    profiler._capture_saved_activation_peak(90)

    assert profiler._saved_activation_peak_allocated == 100
    assert profiler._saved_activation_bytes_at_peak == {"attention": 4096}


def test_memory_breakdown_labels_asym_cpu_adamw_master_and_state_once() -> None:
    name = "model.layers.0.mlp.experts.3.lora_A.default.weight"
    cpu_master = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
    exp_avg = torch.full((2, 2), 0.5, dtype=torch.float32)
    grad_offload_buffer = torch.full((8,), 0.25, dtype=torch.float32)

    class SyntheticAsymCPUAdamW:
        def __init__(self) -> None:
            self.state = {cpu_master: {"cpu_master": cpu_master.data, "exp_avg": exp_avg}}

        def asym_cpu_master_params(self):
            return [cpu_master]

        def asym_cpu_param_name_map(self):
            return {id(cpu_master): name}

        def asym_cpu_adamw_summary(self):
            return {"enabled": True, "backend": "torch"}

        def asym_cpu_adamw_grad_offload_buffer(self):
            return grad_offload_buffer

    profiler = LFMemoryBreakdownProfiler(LFTraceConfig(memory_breakdown=True))
    persistent = profiler._collect_persistent_bytes(None, SyntheticAsymCPUAdamW())

    routed = persistent["routed_experts"]
    assert routed["cpu_master_weight_cpu"] == cpu_master.untyped_storage().nbytes()
    assert routed["optimizer_state_cpu"] == exp_avg.untyped_storage().nbytes()
    assert persistent["optimizer"]["offloaded_grad_cpu"] == grad_offload_buffer.untyped_storage().nbytes()
    assert routed.get("optimizer_state_cpu", 0) != cpu_master.untyped_storage().nbytes() + exp_avg.untyped_storage().nbytes()


def test_source_profile_asym_cpu_adamw_summary_and_warmup_stage_rows() -> None:
    class SyntheticOptimizer:
        def asym_cpu_adamw_summary(self):
            return {
                "enabled": True,
                "backend": "torch",
                "param_count": 2,
                "cpu_master_bytes": 32,
                "optimizer_state_cpu_bytes": 64,
            }

    handle = type("Handle", (), {"optimizer": SyntheticOptimizer(), "prepared_optimizer": SyntheticOptimizer()})()
    summary = lf_profiled_train._asym_cpu_adamw_summary_from_trace(handle)
    assert summary["enabled"] is True
    assert summary["backend"] == "torch"
    assert summary["cpu_master_bytes"] == 32

    recorder = lf_profiled_train.LFProfileRecorder(config={"warmup_steps": 1}, reset_stage_peak_stats=False)
    with recorder.stage("lf.optimizer.step"):
        pass
    with recorder.stage("lf.optimizer.step"):
        pass
    row = next(row for row in recorder._stage_rows() if row["name"] == "lf.optimizer.step")
    assert row["samples"] == 1
    assert row["raw_samples"] == 2
    assert row["warmup_samples_skipped"] == 1


def test_source_profile_reports_activation_offload_counters() -> None:
    class SyntheticExecutionStats:
        def as_dict(self):
            return {
                "asym_forward_calls": 3,
                "asym_dx_calls": 2,
                "expact_lora_a_forward_grouped_calls": 4,
                "expact_lora_a_forward_cpu_left_grouped_calls": 3,
                "expact_lora_a_forward_hbm_grouped_calls": 1,
                "reference_fallback_count": 0,
                "fallback_reasons": {},
            }

    class SyntheticExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.profile_prefix = "layers.0.mlp.experts"
            self.stats = SyntheticExecutionStats()
            self._last_activation_offload_stats = {
                "cpu_owned_bytes": 0,
                "cpu_live_bytes": 0,
                "cpu_peak_bytes_live": 1024,
                "max_stage_bytes_live": 2048,
                "cpu_pool_cached_bytes": 4096,
                "cpu_pool_limit_bytes": 8192,
                "num_offloads": 2,
                "offloaded_bytes": 100,
                "offload_bytes_by_tag": {"X": 64, "S": 36},
                "num_stages": 1,
                "stage_bytes_by_tag": {"S_stage": 32},
                "pre_final_cleanup_cpu_owned_bytes": 0,
                "final_cleanup_released_bytes": 0,
                "source_context": {
                    "num_offloads": 1,
                    "offloaded_bytes": 16,
                    "offload_bytes_by_tag": {"q_proj.U": 16},
                    "num_stages": 0,
                    "stage_bytes_by_tag": {},
                },
            }
            self._last_activation_offload_stats_pre_release = {
                "cpu_owned_bytes": 0,
                "max_stage_bytes_live": 2048,
            }

    class SyntheticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = SyntheticExpert()

    previous = lf_profiled_train._LAST_LF_MODEL
    try:
        lf_profiled_train._LAST_LF_MODEL = SyntheticModel()
        summary = lf_profiled_train._activation_offload_counters_from_model()
    finally:
        lf_profiled_train._LAST_LF_MODEL = previous

    assert summary["available"] is True
    assert summary["module_count"] == 1
    assert summary["total_cpu_live_bytes"] == 0
    assert summary["max_cpu_peak_bytes_live"] == 1024
    assert summary["max_stage_bytes_live"] == 2048
    assert summary["max_cpu_pool_cached_bytes"] == 4096
    assert summary["source_context_count"] == 1
    assert summary["total_d2h_offload_copy_calls"] == 3
    assert summary["total_h2d_stage_copy_calls"] == 1
    assert summary["total_d2h_offloaded_bytes"] == 116
    assert summary["total_h2d_staged_bytes"] == 32
    assert summary["total_forward_offload_copy_calls"] == 3
    assert summary["total_backward_stage_copy_calls"] == 1
    assert summary["total_forward_offloaded_bytes"] == 116
    assert summary["total_backward_staged_bytes"] == 32
    assert summary["total_activation_transfer_calls"] == 4
    assert summary["total_activation_transfer_bytes"] == 148
    assert summary["offload_bytes_by_tag"] == {"X": 64, "S": 36, "q_proj.U": 16}
    assert summary["stage_bytes_by_tag"] == {"S_stage": 32}
    row = summary["rows"][0]
    assert row["name"] == "experts"
    assert row["execution_stats"]["expact_lora_a_forward_cpu_left_grouped_calls"] == 3
    assert row["execution_stats"]["expact_lora_a_forward_hbm_grouped_calls"] == 1
    assert row["execution_stats"]["asym_forward_calls"] == 3
    assert row["activation_offload_stats"]["cpu_pool_limit_bytes"] == 8192
