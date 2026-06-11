from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_profile_launcher_module():
    path = ROOT / "scripts/lf/run_lf_profiled_train.py"
    spec = importlib.util.spec_from_file_location("run_lf_profiled_train_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_profile(
    path: Path,
    *,
    kt: dict,
    lora: dict,
    trainer_log: Path | None = None,
    optimizer_memory: dict | None = None,
    optimizer_memory_preflight: dict | None = None,
    process_memory: dict | None = None,
    stage_memory_rows: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "workload": "unit",
                "config": {
                    "backend": "kt_armbf16",
                    "precision": "bf16",
                    "seq_len": 128,
                    "lora_target": "all",
                    "warmup_steps": 0,
                    "measure_steps": 1,
                },
                "memory": {"gpu": {}, "process": process_memory or {}},
                "trainer": {"trainer_log": str(trainer_log or "")},
                "forward": {"total_milliseconds": 1.0},
                "backward": {"total_milliseconds": 2.0},
                "stage_memory": {"rows": stage_memory_rows or []},
                "kt": kt,
                "lora": lora,
                "optimizer_memory_preflight": optimizer_memory_preflight or {},
                "optimizer_memory": optimizer_memory or {},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_model_capture_emits_startup_kt_lora_summary() -> None:
    module = _load_profile_launcher_module()
    events: list[tuple[str, dict]] = []
    writes: list[tuple[str, dict | None]] = []

    class DummyHeartbeat:
        def emit(self, stage: str, **fields) -> None:
            events.append((stage, fields))

    class DummyPartialWriter:
        def write(self, reason: str, *, force: bool = False, extra: dict | None = None) -> None:
            writes.append((reason, extra))

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A_weight = torch.nn.Parameter(torch.ones(3))
            fused = torch.nn.Parameter(torch.ones(5))
            wrapper = SimpleNamespace(
                method="ARMBF16_SFT",
                _kt_forward_calls=0,
                _kt_backward_calls=0,
                _lora_initialized=True,
                _kt_timing={},
            )
            self._kt_wrappers = [
                SimpleNamespace(
                    layer_idx=0,
                    wrapper=wrapper,
                    _fused_expert_lora_params=torch.nn.ParameterList([fused]),
                )
            ]

    module._MODEL_CAPTURE_HEARTBEAT = DummyHeartbeat()
    module._MODEL_CAPTURE_PARTIAL_WRITER = DummyPartialWriter()
    try:
        model = DummyModel()
        assert module._capture_loaded_model(model) is model
    finally:
        module._MODEL_CAPTURE_HEARTBEAT = None
        module._MODEL_CAPTURE_PARTIAL_WRITER = None

    model_loaded = [fields for stage, fields in events if stage == "model_loaded"]
    assert model_loaded
    assert model_loaded[0]["kt_wrappers"] == 1
    assert model_loaded[0]["trainable_parameters"] == 3
    assert model_loaded[0]["kt_fused_expert_lora_parameters"] == 5
    assert writes and writes[0][0] == "model_loaded"
    assert writes[0][1]["kt"]["wrapper_count"] == 1
    assert writes[0][1]["lora"]["kt_fused_expert_lora_parameters"] == 5


def test_kt_counters_include_native_cache_observability() -> None:
    module = _load_profile_launcher_module()

    wrapper = SimpleNamespace(
        method="ARMBF16_SFT",
        _kt_forward_calls=3,
        _kt_backward_calls=2,
        _lora_initialized=True,
        _cache_depth=1,
        max_cache_depth=2,
        _max_cache_depth_observed=2,
        _buffer=SimpleNamespace(qlen=4096),
        _buffer_allocation_count=2,
        cache_stack_depth=1,
        max_cache_stack_depth_observed=2,
        cache_save_count=3,
        cache_pop_count=2,
        last_cache_entry_bytes=1234,
        total_cache_entry_bytes_saved=5678,
        _kt_timing={},
    )
    model = SimpleNamespace(_kt_wrappers=[SimpleNamespace(layer_idx=7, wrapper=wrapper)])
    old_model = module._LAST_LF_MODEL
    module._LAST_LF_MODEL = model
    try:
        counters = module._kt_counters_from_model()
    finally:
        module._LAST_LF_MODEL = old_model

    row = counters["rows"][0]
    assert row["staging_buffer_scope"] == "wrapper"
    assert row["staging_buffer_capacity_qlen"] == 4096
    assert row["staging_buffer_allocation_count"] == 2
    assert row["native_cache_stack_depth"] == 1
    assert row["native_max_cache_stack_depth_observed"] == 2
    assert row["native_cache_save_count"] == 3
    assert row["native_cache_pop_count"] == 2
    assert row["native_last_cache_entry_bytes"] == 1234
    assert row["native_total_cache_entry_bytes_saved"] == 5678


def test_model_capture_startup_validation_rejects_missing_kt_wrappers(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    events: list[tuple[str, dict]] = []
    writes: list[tuple[str, dict | None]] = []
    monkeypatch.setenv("ASYM_GEMM_LF_REQUIRE_KT_STARTUP", "1")

    class DummyHeartbeat:
        def emit(self, stage: str, **fields) -> None:
            events.append((stage, fields))

    class DummyPartialWriter:
        def write(self, reason: str, *, force: bool = False, extra: dict | None = None) -> None:
            writes.append((reason, extra))

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3))

    module._MODEL_CAPTURE_HEARTBEAT = DummyHeartbeat()
    module._MODEL_CAPTURE_PARTIAL_WRITER = DummyPartialWriter()
    try:
        with pytest.raises(RuntimeError, match="KT startup counters unavailable"):
            module._capture_loaded_model(DummyModel())
    finally:
        module._MODEL_CAPTURE_HEARTBEAT = None
        module._MODEL_CAPTURE_PARTIAL_WRITER = None

    assert any(stage == "model_loaded" for stage, _ in events)
    failures = [fields for stage, fields in events if stage == "model_loaded_startup_validation_failed"]
    assert failures and "KT startup counters unavailable" in failures[0]["error"]
    assert writes[-1][0] == "model_loaded_startup_validation_failed"
    assert "KT startup counters unavailable" in writes[-1][1]["error"]


def test_model_capture_startup_validation_rejects_missing_fused_lora(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    events: list[tuple[str, dict]] = []
    writes: list[tuple[str, dict | None]] = []
    monkeypatch.setenv("ASYM_GEMM_LF_REQUIRE_KT_STARTUP", "1")
    monkeypatch.setenv("ASYM_GEMM_LF_REQUIRE_KT_FUSED_LORA_STARTUP", "1")

    class DummyHeartbeat:
        def emit(self, stage: str, **fields) -> None:
            events.append((stage, fields))

    class DummyPartialWriter:
        def write(self, reason: str, *, force: bool = False, extra: dict | None = None) -> None:
            writes.append((reason, extra))

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A_weight = torch.nn.Parameter(torch.ones(3))
            wrapper = SimpleNamespace(
                method="ARMBF16_SFT",
                _kt_forward_calls=0,
                _kt_backward_calls=0,
                _lora_initialized=True,
                _kt_timing={},
            )
            self._kt_wrappers = [SimpleNamespace(layer_idx=0, wrapper=wrapper)]

    module._MODEL_CAPTURE_HEARTBEAT = DummyHeartbeat()
    module._MODEL_CAPTURE_PARTIAL_WRITER = DummyPartialWriter()
    try:
        with pytest.raises(RuntimeError, match="KT fused expert LoRA params must be positive"):
            module._capture_loaded_model(DummyModel())
    finally:
        module._MODEL_CAPTURE_HEARTBEAT = None
        module._MODEL_CAPTURE_PARTIAL_WRITER = None

    failures = [fields for stage, fields in events if stage == "model_loaded_startup_validation_failed"]
    assert failures
    assert "KT fused expert LoRA params must be positive" in failures[0]["error"]
    assert writes[-1][0] == "model_loaded_startup_validation_failed"
    assert writes[-1][1]["kt"]["wrapper_count"] == 1


def test_profile_config_records_launch_shape_aliases_and_triton_cache(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    monkeypatch.setenv("ASYM_GEMM_LF_CONFIG_BACKEND", "kt_armbf16")
    monkeypatch.setenv("ASYM_GEMM_LF_CONFIG_SEQ_LEN", "128")
    monkeypatch.setenv("ASYM_GEMM_LF_CONFIG_LOGICAL_QLEN", "256")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/asymgemm_triton_cache/unit")

    config = module._config_from_args(
        [
            "--model_name_or_path",
            "unit/unit",
            "--dataset",
            "dummy",
            "--template",
            "qwen3_nothink",
            "--cutoff_len",
            "128",
            "--per_device_train_batch_size",
            "2",
            "--gradient_accumulation_steps",
            "1",
            "--lora_rank",
            "8",
            "--max_steps",
            "1",
        ]
    )

    assert config["batch_size"] == 2
    assert config["per_device_train_batch_size"] == 2
    assert config["seq_len"] == 128
    assert config["logical_qlen"] == 256
    assert config["gradient_accumulation_steps"] == 1
    assert config["triton_cache_dir"] == "/tmp/asymgemm_triton_cache/unit"
    affinity = config["cpu_affinity"]
    assert isinstance(affinity, dict)
    if affinity["available"]:
        assert affinity["count"] >= 1
        assert isinstance(affinity["cpus"], str)
        assert config["cpu_affinity_count"] == affinity["count"]


def test_kt_fused_lora_update_health_detects_sampled_param_change() -> None:
    module = _load_profile_launcher_module()

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            fused = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
            fused.grad = torch.ones_like(fused)
            self._fused = fused
            self._kt_wrappers = [
                SimpleNamespace(
                    layer_idx=0,
                    _fused_expert_lora_params=torch.nn.ParameterList([fused]),
                )
            ]

    model = DummyModel()
    before = module._kt_fused_lora_update_snapshot(model)
    with torch.no_grad():
        model._fused.add_(model._fused.grad, alpha=-0.1)
    after = module._kt_fused_lora_update_snapshot(model)
    health = module._kt_fused_lora_update_health(before, after)

    assert health["available"] is True
    assert health["sampled_tensors"] == 1
    assert health["total_fused_tensors"] == 1
    assert health["exhaustive"] is True
    assert health["grad_nonzero_tensors"] == 1
    assert health["changed_tensors"] == 1
    assert health["updated_grad_tensors"] == 1
    assert health["grad_nonzero_unchanged_tensors"] == 0
    assert health["passed"] is True
    assert health["rows"][0]["grad_nonzero_before_step"] is True
    assert health["rows"][0]["param_changed_after_step"] is True
    assert health["rows"][0]["nonzero_grad_changed_after_step"] is True


def test_kt_fused_lora_update_health_can_check_all_tensors(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    monkeypatch.setenv("ASYM_GEMM_LF_KT_LORA_HEALTH_MAX_TENSORS", "all")

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            params = [torch.nn.Parameter(torch.arange(index + 1, index + 9, dtype=torch.float32)) for index in range(3)]
            for param in params:
                param.grad = torch.ones_like(param)
            self._params = torch.nn.ParameterList(params)
            self._kt_wrappers = [
                SimpleNamespace(
                    layer_idx=0,
                    _fused_expert_lora_params=torch.nn.ParameterList(params[:2]),
                ),
                SimpleNamespace(
                    layer_idx=1,
                    _fused_expert_lora_params=torch.nn.ParameterList(params[2:]),
                ),
            ]

    model = DummyModel()
    before = module._kt_fused_lora_update_snapshot(model)
    with torch.no_grad():
        for param in model._params:
            param.add_(param.grad, alpha=-0.1)
    after = module._kt_fused_lora_update_snapshot(model)
    health = module._kt_fused_lora_update_health(before, after)

    assert before["requested_max_tensors"] == "all"
    assert before["sampled_tensors"] == 3
    assert before["total_fused_tensors"] == 3
    assert before["exhaustive"] is True
    assert health["passed"] is True
    assert health["exhaustive"] is True
    assert health["updated_grad_tensors"] == 3
    assert health["grad_nonzero_unchanged_tensors"] == 0
    assert "all nonzero-gradient fused LoRA tensors changed" in health["reason"]


def test_sample_tensor_stats_spreads_sample_to_tail() -> None:
    module = _load_profile_launcher_module()
    tensor = torch.zeros(4097, dtype=torch.float32)
    tensor[-1] = 1.0

    stats = module._sample_tensor_stats(tensor, max_samples=4096)

    assert stats["sample_all_elements"] is False
    assert stats["sample_count"] == 4096
    assert stats["sample_abs_sum"] == 1.0
    assert stats["sample_weighted_sum"] == 4096.0


def test_kt_fused_lora_update_health_can_check_all_tensor_elements(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    monkeypatch.setenv("ASYM_GEMM_LF_KT_LORA_HEALTH_MAX_TENSORS", "all")
    monkeypatch.setenv("ASYM_GEMM_LF_KT_LORA_HEALTH_MAX_ELEMENTS", "all")

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            fused = torch.nn.Parameter(torch.arange(32, dtype=torch.float32))
            fused.grad = torch.ones_like(fused)
            self._fused = fused
            self._kt_wrappers = [
                SimpleNamespace(
                    layer_idx=0,
                    _fused_expert_lora_params=torch.nn.ParameterList([fused]),
                )
            ]

    model = DummyModel()
    before = module._kt_fused_lora_update_snapshot(model)
    with torch.no_grad():
        model._fused.add_(model._fused.grad, alpha=-0.1)
    after = module._kt_fused_lora_update_snapshot(model)
    health = module._kt_fused_lora_update_health(before, after)

    assert before["requested_max_elements"] == "all"
    assert before["exhaustive_elements"] is True
    assert before["rows"][0]["param"]["sample_count"] == before["rows"][0]["param"]["numel"]
    assert before["rows"][0]["param"]["sample_all_elements"] is True
    assert health["passed"] is True
    assert health["exhaustive_elements"] is True
    assert "full-element checksums" in health["reason"]


def test_kt_fused_lora_update_health_fails_when_tensor_set_changes() -> None:
    module = _load_profile_launcher_module()
    before = {
        "available": True,
        "sampled_tensors": 2,
        "total_fused_tensors": 2,
        "exhaustive": True,
        "rows": [
            {
                "layer_idx": 0,
                "layer_index": 0,
                "param_index": 0,
                "param": {"sample_sum": 1.0, "sample_abs_sum": 1.0, "sample_weighted_sum": 1.0, "sample_max_abs": 1.0},
                "grad": {"sample_abs_sum": 1.0},
            },
            {
                "layer_idx": 0,
                "layer_index": 0,
                "param_index": 1,
                "param": {"sample_sum": 2.0, "sample_abs_sum": 2.0, "sample_weighted_sum": 2.0, "sample_max_abs": 2.0},
                "grad": {"sample_abs_sum": 1.0},
            },
        ],
    }
    after = {
        "available": True,
        "sampled_tensors": 1,
        "total_fused_tensors": 1,
        "exhaustive": True,
        "rows": [
            {
                "layer_idx": 0,
                "layer_index": 0,
                "param_index": 0,
                "param": {"sample_sum": 0.9, "sample_abs_sum": 0.9, "sample_weighted_sum": 0.9, "sample_max_abs": 0.9},
            }
        ],
    }

    health = module._kt_fused_lora_update_health(before, after)

    assert health["passed"] is False
    assert health["missing_after_tensors"] == 1
    assert "tensor set changed" in health["reason"]


def test_kt_optimizer_memory_preflight_estimates_adamw_policies() -> None:
    module = _load_profile_launcher_module()
    lora = {
        "available": True,
        "trainable_parameters": 10,
        "peft_lora_parameters": 4,
        "kt_peft_expert_lora_parameters": 1,
        "kt_expert_lora_parameters": 6,
        "kt_fused_expert_lora_parameters": 6,
    }

    preflight = module._kt_optimizer_memory_preflight(lora, {"logical_qlen": 256, "lora_rank": 8})

    assert preflight["available"] is True
    assert preflight["trainable_parameters"] == 10
    assert preflight["kt_fused_expert_lora_parameters"] == 6
    assert preflight["non_expert_peft_lora_parameters"] == 3
    assert preflight["param_bf16_bytes"] == 20
    assert preflight["grad_bf16_bytes"] == 20
    assert preflight["adamw_bf16_moments_bytes"] == 40
    assert preflight["adamw_fp32_moments_bytes"] == 80
    assert preflight["adamw_fp32_master_bytes"] == 40
    assert preflight["total_bf16_params_grads_adamw_bf16_moments_bytes"] == 80
    assert preflight["total_bf16_params_grads_adamw_fp32_moments_master_bytes"] == 160
    assert preflight["logical_qlen"] == 256
    assert preflight["lora_rank"] == 8


def test_kt_optimizer_memory_preflight_rejects_missing_trainable_counter() -> None:
    module = _load_profile_launcher_module()

    preflight = module._kt_optimizer_memory_preflight({"available": True}, {"logical_qlen": 256, "lora_rank": 8})

    assert preflight["available"] is False
    assert "trainable_parameters" in preflight["reason"]


def test_process_memory_snapshot_reports_rss_when_procfs_available() -> None:
    module = _load_profile_launcher_module()

    snapshot = module._process_memory_snapshot()

    assert "available" in snapshot
    if snapshot["available"]:
        assert snapshot["rss_bytes"] > 0
        assert snapshot["virtual_memory_bytes"] >= snapshot["rss_bytes"]


def test_optimizer_step_hook_brackets_rss_around_original_step(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    events: list[tuple[str, dict]] = []
    writes: list[tuple[str, dict | None]] = []

    class DummyHeartbeat:
        def emit(self, stage: str, **fields) -> None:
            events.append((stage, fields))

    class DummyPartialWriter:
        def write(self, reason: str, *, force: bool = False, extra: dict | None = None) -> None:
            writes.append((reason, extra))

    class DummyTrainer:
        def __init__(self) -> None:
            self.state = SimpleNamespace(global_step=7)
            self.model = torch.nn.Linear(1, 1, bias=False)
            self.optimizer = None

        def train(self, *args, **kwargs):
            return None

        def get_batch_samples(self, *args, **kwargs):
            return []

        def training_step(self, model, inputs, *args, **kwargs):
            return torch.tensor(0.0)

        def _clip_grad_norm(self, model):
            return None

        def create_optimizer(self):
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            return self.optimizer

    fake_transformers = ModuleType("transformers")
    fake_transformers.Trainer = DummyTrainer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    snapshots = iter(
        [
            {"available": True, "rss_bytes": 100, "rss_peak_bytes": 100, "virtual_memory_bytes": 1000},
            {"available": True, "rss_bytes": 110, "rss_peak_bytes": 110, "virtual_memory_bytes": 1000},
            {"available": True, "rss_bytes": 210, "rss_peak_bytes": 210, "virtual_memory_bytes": 1000},
        ]
    )
    monkeypatch.setattr(module, "_process_memory_snapshot", lambda: next(snapshots))
    module._OPTIMIZER_MEMORY_MARKER = None
    module._install_trainer_heartbeat_hooks(DummyHeartbeat(), DummyPartialWriter())

    trainer = DummyTrainer()
    optimizer = trainer.create_optimizer()
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()

    summary = module._OPTIMIZER_MEMORY_MARKER
    assert summary["process_memory_at_start"]["rss_bytes"] == 100
    assert summary["process_memory_before_step"]["rss_bytes"] == 110
    assert summary["process_memory_after_step"]["rss_bytes"] == 210
    assert summary["process_rss_pre_step_overhead_delta_bytes"] == 10
    assert summary["process_rss_delta_bytes"] == 100
    assert [stage for stage, _ in events] == ["optimizer_step_start", "optimizer_step_end"]
    assert [reason for reason, _ in writes] == ["optimizer_step_start", "optimizer_step_end"]


def test_optimizer_step_hook_writes_exception_memory_evidence(monkeypatch) -> None:
    module = _load_profile_launcher_module()
    events: list[tuple[str, dict]] = []
    writes: list[tuple[str, dict | None]] = []

    class DummyHeartbeat:
        def emit(self, stage: str, **fields) -> None:
            events.append((stage, fields))

    class DummyPartialWriter:
        def write(self, reason: str, *, force: bool = False, extra: dict | None = None) -> None:
            writes.append((reason, extra))

    class RaisingOptimizer:
        state: dict = {}

        def step(self):
            raise RuntimeError("lazy optimizer allocation failed")

    class DummyTrainer:
        def __init__(self) -> None:
            self.state = SimpleNamespace(global_step=3)
            self.model = torch.nn.Linear(1, 1, bias=False)
            self.optimizer = None

        def train(self, *args, **kwargs):
            return None

        def get_batch_samples(self, *args, **kwargs):
            return []

        def training_step(self, model, inputs, *args, **kwargs):
            return torch.tensor(0.0)

        def _clip_grad_norm(self, model):
            return None

        def create_optimizer(self):
            self.optimizer = RaisingOptimizer()
            return self.optimizer

    fake_transformers = ModuleType("transformers")
    fake_transformers.Trainer = DummyTrainer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    snapshots = iter(
        [
            {"available": True, "rss_bytes": 500, "rss_peak_bytes": 500, "virtual_memory_bytes": 1000},
            {"available": True, "rss_bytes": 550, "rss_peak_bytes": 550, "virtual_memory_bytes": 1000},
            {"available": True, "rss_bytes": 700, "rss_peak_bytes": 700, "virtual_memory_bytes": 1000},
        ]
    )
    monkeypatch.setattr(module, "_process_memory_snapshot", lambda: next(snapshots))
    module._OPTIMIZER_MEMORY_MARKER = None
    module._install_trainer_heartbeat_hooks(DummyHeartbeat(), DummyPartialWriter())

    trainer = DummyTrainer()
    optimizer = trainer.create_optimizer()
    with pytest.raises(RuntimeError, match="lazy optimizer allocation failed"):
        optimizer.step()

    summary = module._OPTIMIZER_MEMORY_MARKER
    assert summary["exception_type"] == "RuntimeError"
    assert summary["process_memory_before_step"]["rss_bytes"] == 550
    assert summary["process_memory_after_step"]["rss_bytes"] == 700
    assert summary["process_rss_delta_bytes"] == 150
    assert summary["kt_lora_update_health"]["passed"] is False
    assert [stage for stage, _ in events] == ["optimizer_step_start", "optimizer_step_exception"]
    assert [reason for reason, _ in writes] == ["optimizer_step_start", "optimizer_step_exception"]


def _run_postprocess(tmp_path: Path, *, kt: dict, lora: dict, optimizer_memory: dict | None = None) -> str:
    source_profile = tmp_path / "source_profile.json"
    output_dir = tmp_path / "out"
    _write_source_profile(source_profile, kt=kt, lora=lora, optimizer_memory=optimizer_memory)
    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--source-profile-json",
            str(source_profile),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    return (output_dir / "summary.md").read_text(encoding="utf-8")


def _run_postprocess_output(tmp_path: Path, *, kt: dict, lora: dict) -> Path:
    source_profile = tmp_path / "source_profile.json"
    output_dir = tmp_path / "out"
    _write_source_profile(source_profile, kt=kt, lora=lora)
    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--source-profile-json",
            str(source_profile),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    return output_dir


def _run_postprocess_output_with_optimizer_memory(
    tmp_path: Path,
    *,
    kt: dict,
    lora: dict,
    optimizer_memory: dict,
) -> Path:
    source_profile = tmp_path / "source_profile.json"
    output_dir = tmp_path / "out"
    _write_source_profile(source_profile, kt=kt, lora=lora, optimizer_memory=optimizer_memory)
    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--source-profile-json",
            str(source_profile),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    return output_dir


def _run_postprocess_output_with_optimizer_preflight(
    tmp_path: Path,
    *,
    kt: dict,
    lora: dict,
    optimizer_memory_preflight: dict,
) -> Path:
    source_profile = tmp_path / "source_profile.json"
    output_dir = tmp_path / "out"
    _write_source_profile(
        source_profile,
        kt=kt,
        lora=lora,
        optimizer_memory_preflight=optimizer_memory_preflight,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--source-profile-json",
            str(source_profile),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    return output_dir


def _run_postprocess_with_log(tmp_path: Path, *, kt: dict, lora: dict, log_text: str) -> str:
    source_profile = tmp_path / "source_profile.json"
    output_dir = tmp_path / "out"
    train_log = tmp_path / "train.log"
    train_log.write_text(log_text, encoding="utf-8")
    _write_source_profile(source_profile, kt=kt, lora=lora, trainer_log=train_log)
    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--source-profile-json",
            str(source_profile),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    return (output_dir / "summary.md").read_text(encoding="utf-8")


def test_source_summary_renders_missing_lora_counters_as_missing(tmp_path: Path) -> None:
    summary = _run_postprocess(
        tmp_path,
        kt={"available": False, "reason": "model hook did not capture a model"},
        lora={"available": False, "reason": "model hook did not capture a model"},
    )

    assert "| KT wrappers | missing (model hook did not capture a model) |" in summary
    assert "| trainable params | missing (model hook did not capture a model) |" in summary
    assert "| KT fused expert LoRA params | missing (model hook did not capture a model) |" in summary


def test_source_summary_preserves_real_zero_lora_counters(tmp_path: Path) -> None:
    summary = _run_postprocess(
        tmp_path,
        kt={"available": True, "wrapper_count": 0, "total_forward_calls": 0, "total_backward_calls": 0},
        lora={
            "available": True,
            "trainable_parameters": 0,
            "peft_lora_parameters": 0,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 0,
            "kt_peft_expert_lora_parameters": 0,
            "kt_fused_expert_lora_parameters": 0,
        },
    )

    assert "| KT wrappers | 0 |" in summary
    assert "| trainable params | 0 |" in summary
    assert "| KT fused expert LoRA params | 0 |" in summary


def test_source_summary_treats_incomplete_lora_surface_as_unknown(tmp_path: Path) -> None:
    output_dir = _run_postprocess_output(
        tmp_path,
        kt={"available": True, "wrapper_count": 1, "total_forward_calls": 0, "total_backward_calls": 0},
        lora={"available": True, "trainable_parameters": 10},
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    surface_csv = (output_dir / "trainable_surface.csv").read_text(encoding="utf-8")
    assert "| trainable surface | unknown |" in summary
    assert "missing LoRA surface counters" in surface_csv
    assert "no trainable LoRA detected" not in summary


def test_source_summary_flags_kt_attention_plus_expert_surface(tmp_path: Path) -> None:
    output_dir = _run_postprocess_output(
        tmp_path,
        kt={"available": True, "wrapper_count": 48, "total_forward_calls": 1, "total_backward_calls": 0},
        lora={
            "available": True,
            "trainable_parameters": 3_375_366_144,
            "peft_lora_parameters": 53_477_376,
            "lf_fused_expert_lora_parameters": 0,
            "lf_fused_expert_lora_tensors": 0,
            "kt_expert_lora_parameters": 3_321_888_768,
            "kt_peft_expert_lora_parameters": 0,
            "kt_fused_expert_lora_tensors": 288,
            "kt_fused_expert_lora_parameters": 3_321_888_768,
        },
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    profile = json.loads((output_dir / "profile.json").read_text(encoding="utf-8"))
    surface_csv = (output_dir / "trainable_surface.csv").read_text(encoding="utf-8")
    assert "| trainable surface | attention+expert LoRA |" in summary
    assert "| non-expert PEFT LoRA params | 53477376 |" in summary
    assert "| expert LoRA params | 3321888768 |" in summary
    assert "| KT fused expert LoRA tensors | 288 |" in summary
    assert "| KT fused expert LoRA sidecar expected tensors | 288 |" in summary
    assert "| KT fused expert LoRA sidecar expected params | 3321888768 |" in summary
    assert "| backend comparison note | requires a baseline that also trains expert LoRA |" in summary
    assert profile["trainable_surface"]["surface"] == "attention+expert LoRA"
    assert profile["trainable_surface"]["expert_lora_parameters"] == 3_321_888_768
    assert "attention+expert LoRA" in surface_csv
    assert "requires a baseline that also trains expert LoRA" in surface_csv


def test_source_summary_flags_attention_only_surface(tmp_path: Path) -> None:
    summary = _run_postprocess(
        tmp_path,
        kt={"available": True, "wrapper_count": 0, "total_forward_calls": 0, "total_backward_calls": 0},
        lora={
            "available": True,
            "trainable_parameters": 53_477_376,
            "peft_lora_parameters": 53_477_376,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 0,
            "kt_peft_expert_lora_parameters": 0,
            "kt_fused_expert_lora_parameters": 0,
        },
    )

    assert "| trainable surface | attention-only LoRA |" in summary
    assert "| expert LoRA params | 0 |" in summary
    assert (
        "| backend comparison note | attention-only surface; do not compare directly to expert-LoRA KT runs |"
        in summary
    )


def test_source_summary_uses_trainer_log_trainable_fallback(tmp_path: Path) -> None:
    summary = _run_postprocess_with_log(
        tmp_path,
        kt={"available": True, "wrapper_count": 0, "total_forward_calls": 0, "total_backward_calls": 0},
        lora={"available": False, "reason": "model hook did not capture a model"},
        log_text=(
            "trainable params: 53,477,376 || all params: 30,577,000,000 || trainable%: 0.1749\n"
            "[INFO|trainer.py:1479] >>   Number of trainable parameters = 53,477,376\n"
        ),
    )

    assert "| trainable params | 53477376 (trainer log fallback) |" in summary
    assert "| trainable surface | unknown |" in summary
    assert "| backend comparison note | only trainer-log trainable parameter count is available |" in summary


def test_source_summary_and_csv_include_kt_lora_update_health(tmp_path: Path) -> None:
    output_dir = _run_postprocess_output_with_optimizer_memory(
        tmp_path,
        kt={"available": True, "wrapper_count": 1, "total_forward_calls": 2, "total_backward_calls": 1},
        lora={
            "available": True,
            "trainable_parameters": 8,
            "peft_lora_parameters": 0,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 8,
            "kt_peft_expert_lora_parameters": 0,
            "kt_fused_expert_lora_parameters": 8,
        },
        optimizer_memory={
            "kt_lora_update_health": {
                "available": True,
                "sampled_tensors": 1,
                "total_fused_tensors": 1,
                "after_sampled_tensors": 1,
                "after_total_fused_tensors": 1,
                "requested_max_tensors": "all",
                "requested_max_elements": "all",
                "exhaustive_elements": True,
                "compared_tensors": 1,
                "missing_after_tensors": 0,
                "unexpected_after_tensors": 0,
                "grad_nonzero_tensors": 1,
                "changed_tensors": 1,
                "rows": [
                    {
                        "layer_idx": 0,
                        "param_index": 0,
                        "grad_nonzero_before_step": True,
                        "param_changed_after_step": True,
                    }
                ],
            }
        },
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "kt_lora_update_health.csv").read_text(encoding="utf-8")
    assert "## KT Fused LoRA Update Health" in summary
    assert "| passed | True |" in summary
    assert "| exhaustive | True |" in summary
    assert "| exhaustive elements | True |" in summary
    assert "| after total fused tensors | 1 |" in summary
    assert "| missing after tensors | 0 |" in summary
    assert "| requested max tensors | all |" in summary
    assert "| requested max elements | all |" in summary
    assert "| nonzero-gradient tensors changed | 1 |" in summary
    assert "after_total_fused_tensors" in csv_text
    assert "missing_after_tensors" in csv_text
    assert "all nonzero-gradient fused LoRA tensors changed after optimizer step" in csv_text


def test_source_summary_recomputes_stale_kt_lora_update_health_fail_closed(tmp_path: Path) -> None:
    output_dir = _run_postprocess_output_with_optimizer_memory(
        tmp_path,
        kt={"available": True, "wrapper_count": 1, "total_forward_calls": 2, "total_backward_calls": 1},
        lora={
            "available": True,
            "trainable_parameters": 8,
            "peft_lora_parameters": 0,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 8,
            "kt_peft_expert_lora_parameters": 0,
            "kt_fused_expert_lora_parameters": 8,
        },
        optimizer_memory={
            "kt_lora_update_health": {
                "available": True,
                "passed": True,
                "reason": "stale pass from an older profiler",
                "sampled_tensors": 1,
                "total_fused_tensors": 1,
                "after_sampled_tensors": 1,
                "after_total_fused_tensors": 1,
                "compared_tensors": 0,
                "grad_nonzero_tensors": 0,
                "rows": [],
            }
        },
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "kt_lora_update_health.csv").read_text(encoding="utf-8")
    profile = json.loads((output_dir / "profile.json").read_text(encoding="utf-8"))
    health = profile["optimizer_memory"]["kt_lora_update_health"]
    assert "| passed | False |" in summary
    assert "exhaustive fused LoRA health compared 0 of 1 tensors" in summary
    assert health["input_passed"] is True
    assert health["passed"] is False
    assert "input_passed" in csv_text
    assert "stale pass from an older profiler" in csv_text


def test_source_summary_reports_kt_lora_tensor_set_mismatch_fields(tmp_path: Path) -> None:
    output_dir = _run_postprocess_output_with_optimizer_memory(
        tmp_path,
        kt={"available": True, "wrapper_count": 1, "total_forward_calls": 2, "total_backward_calls": 1},
        lora={
            "available": True,
            "trainable_parameters": 8,
            "peft_lora_parameters": 0,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 8,
            "kt_peft_expert_lora_parameters": 0,
            "kt_fused_expert_lora_parameters": 8,
        },
        optimizer_memory={
            "kt_lora_update_health": {
                "available": True,
                "sampled_tensors": 2,
                "total_fused_tensors": 2,
                "after_sampled_tensors": 1,
                "after_total_fused_tensors": 2,
                "compared_tensors": 1,
                "missing_after_tensors": 1,
                "unexpected_after_tensors": 0,
                "grad_nonzero_tensors": 1,
                "rows": [
                    {
                        "layer_idx": 0,
                        "param_index": 0,
                        "grad_nonzero_before_step": True,
                        "param_changed_after_step": True,
                    }
                ],
            }
        },
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "kt_lora_update_health.csv").read_text(encoding="utf-8")
    assert "| passed | False |" in summary
    assert "| after sampled tensors | 1 |" in summary
    assert "| missing after tensors | 1 |" in summary
    assert "sampled fused LoRA tensor set changed" in summary
    assert "missing_after_tensors" in csv_text
    assert "after_sampled_tensors" in csv_text


def test_source_summary_and_csv_include_optimizer_memory_preflight(tmp_path: Path) -> None:
    output_dir = _run_postprocess_output_with_optimizer_preflight(
        tmp_path,
        kt={"available": True, "wrapper_count": 1, "total_forward_calls": 0, "total_backward_calls": 0},
        lora={
            "available": True,
            "trainable_parameters": 10,
            "peft_lora_parameters": 4,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 6,
            "kt_peft_expert_lora_parameters": 1,
            "kt_fused_expert_lora_parameters": 6,
        },
        optimizer_memory_preflight={
            "available": True,
            "source": "lora_counters_pre_optimizer_step",
            "reason": "estimated before lazy optimizer state allocation",
            "assumed_param_dtype": "bf16",
            "logical_qlen": 256,
            "lora_rank": 8,
            "trainable_parameters": 10,
            "kt_fused_expert_lora_parameters": 6,
            "kt_expert_lora_parameters": 6,
            "non_expert_peft_lora_parameters": 3,
            "param_bf16_bytes": 20,
            "grad_bf16_bytes": 20,
            "adamw_bf16_moments_bytes": 40,
            "adamw_fp32_moments_bytes": 80,
            "adamw_fp32_master_bytes": 40,
            "total_bf16_params_grads_adamw_bf16_moments_bytes": 80,
            "total_bf16_params_grads_adamw_fp32_moments_bytes": 120,
            "total_bf16_params_grads_adamw_fp32_moments_master_bytes": 160,
            "large_surface_warning": False,
        },
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "optimizer_memory_preflight.csv").read_text(encoding="utf-8")
    assert "## Optimizer Memory Preflight" in summary
    assert "| trainable params | 10 |" in summary
    assert "| reason | estimated before lazy optimizer state allocation |" in summary
    assert "| assumed param dtype | bf16 |" in summary
    assert "| logical qlen | 256 |" in summary
    assert "| LoRA rank | 8 |" in summary
    assert "| non-expert PEFT LoRA params | 3 |" in summary
    assert "| total with FP32 moments + master bytes | 160 |" in summary
    assert "lora_counters_pre_optimizer_step" in csv_text


def test_source_summary_and_csv_include_process_memory(tmp_path: Path) -> None:
    source_profile = tmp_path / "source_profile.json"
    output_dir = tmp_path / "out"
    _write_source_profile(
        source_profile,
        kt={"available": True, "wrapper_count": 1, "total_forward_calls": 0, "total_backward_calls": 0},
        lora={"available": True, "trainable_parameters": 10},
        process_memory={
            "available": True,
            "source": "/proc/self/status",
            "rss_bytes": 1048576,
            "rss_peak_bytes": 2097152,
            "virtual_memory_bytes": 4194304,
        },
        stage_memory_rows=[
            {
                "name": "step.forward",
                "samples": 1,
                "avg_process_rss_start_bytes": 1048576,
                "avg_process_rss_end_bytes": 1572864,
                "avg_process_rss_delta_bytes": 524288,
                "max_process_rss_peak_end_bytes": 2097152,
            }
        ],
        optimizer_memory={
            "process_memory_before_step": {"available": True, "rss_bytes": 1572864},
            "process_memory_after_step": {"available": True, "rss_bytes": 3145728},
            "process_rss_delta_bytes": 1572864,
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--source-profile-json",
            str(source_profile),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    csv_text = (output_dir / "process_memory.csv").read_text(encoding="utf-8")
    assert "## Process Memory" in summary
    assert "| RSS bytes | 1048576 |" in summary
    assert "| step.forward | 1572864 | 2097152 | - | 524288 |" in summary
    assert "| optimizer_step_after | 3145728 | - | - | 1572864 |" in summary
    assert "step.forward" in csv_text
    assert "optimizer_step_after" in csv_text
    assert "process_rss_delta_bytes" in csv_text


def test_profile_json_csvs_include_nested_source_profile_artifacts(tmp_path: Path) -> None:
    source_profile = {
        "workload": "unit",
        "config": {"backend": "kt_armbf16", "precision": "bf16", "seq_len": 128, "lora_target": "all"},
        "memory": {"gpu": {}},
        "trainer": {},
        "forward": {"total_milliseconds": 1.0},
        "backward": {"total_milliseconds": 2.0},
        "stage_memory": {"rows": []},
        "kt": {
            "available": True,
            "wrapper_count": 1,
            "total_forward_calls": 2,
            "total_backward_calls": 1,
            "rows": [{"layer_idx": 0, "method": "ARMBF16_SFT"}],
        },
        "lora": {
            "available": True,
            "trainable_parameters": 10,
            "peft_lora_parameters": 4,
            "lf_fused_expert_lora_parameters": 0,
            "kt_expert_lora_parameters": 6,
            "kt_peft_expert_lora_parameters": 1,
            "kt_fused_expert_lora_parameters": 6,
        },
        "optimizer_memory_preflight": {
            "available": True,
            "source": "lora_counters_pre_optimizer_step",
            "trainable_parameters": 10,
        },
        "optimizer_memory": {
            "kt_lora_update_health": {
                "available": True,
                "passed": True,
                "sampled_tensors": 1,
                "total_fused_tensors": 1,
                "grad_nonzero_tensors": 1,
                "rows": [
                    {
                        "layer_idx": 0,
                        "param_index": 0,
                        "grad_nonzero_before_step": True,
                        "param_changed_after_step": True,
                    }
                ],
            }
        },
    }
    profile_json = tmp_path / "profile.json"
    output_dir = tmp_path / "out"
    profile_json.write_text(json.dumps({"source_profile": source_profile, "stages": []}) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/lf/postprocess_lf_profile_artifacts.py",
            "--profile-json",
            str(profile_json),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    assert "ARMBF16_SFT" in (output_dir / "kt_counters.csv").read_text(encoding="utf-8")
    assert "attention+expert LoRA" in (output_dir / "trainable_surface.csv").read_text(encoding="utf-8")
    assert "lora_counters_pre_optimizer_step" in (output_dir / "optimizer_memory_preflight.csv").read_text(
        encoding="utf-8"
    )
    assert "param_changed_after_step" in (output_dir / "kt_lora_update_health.csv").read_text(encoding="utf-8")
