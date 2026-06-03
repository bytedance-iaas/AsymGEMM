from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from asym_gemm.training.kt_moe import KTBackendUnavailable, _import_kt_moe_wrapper


ROOT = Path(__file__).resolve().parents[2]
PROFILE_LORA_E2E_PATH = ROOT / "scripts" / "lora" / "profile_lora_e2e.py"
PROFILE_LORA_E2E_DRIVER_PATH = ROOT / "scripts" / "lora" / "profile_lora_e2e_driver.py"
POSTPROCESS_NSYS_LORA_PATH = ROOT / "scripts" / "lora" / "postprocess_nsys_lora.py"
PLOT_ACTIVATION_SWEEP_PATH = ROOT / "scripts" / "plotting" / "plot_activation_recompute_sweep.py"


def _load_profile_lora_e2e_module():
    spec = importlib.util.spec_from_file_location("profile_lora_e2e_under_test", PROFILE_LORA_E2E_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_profile_lora_e2e_driver_module():
    spec = importlib.util.spec_from_file_location("profile_lora_e2e_driver_under_test", PROFILE_LORA_E2E_DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_postprocess_nsys_lora_module():
    spec = importlib.util.spec_from_file_location("postprocess_nsys_lora_under_test", POSTPROCESS_NSYS_LORA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_activation_sweep_plot_module():
    spec = importlib.util.spec_from_file_location("plot_activation_recompute_sweep_under_test", PLOT_ACTIVATION_SWEEP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_lora_e2e_public_backends_are_canonical() -> None:
    profile_lora = _load_profile_lora_e2e_module()

    assert profile_lora.BACKEND_CHOICES == ("torch", "asym", "kt")


def test_profile_lora_e2e_rejects_expert_threshold_for_non_moe_workload() -> None:
    profile_lora = _load_profile_lora_e2e_module()
    args = argparse.Namespace(
        backend="torch",
        workload="dense",
        lora_dtype="bf16",
        expert_recompute_policy="tok",
        expert_recompute_threshold=8,
        expert_recompute_token_min=1,
        expert_recompute_token_max=8,
        expert_activation_save_policy="save_all",
        expert_activation_save_threshold=0,
        expert_activation_save_token_min=1,
        expert_activation_save_token_max=None,
    )

    with pytest.raises(ValueError, match="only supported for MoE"):
        profile_lora.validate_backend_workload(args)


def test_profile_lora_e2e_public_workload_names_are_blockwise_for_moe() -> None:
    profile_lora = _load_profile_lora_e2e_module()

    assert "dense_14b" in profile_lora.WORKLOAD_CHOICES
    assert "moe-604m-a75m" in profile_lora.WORKLOAD_CHOICES
    assert "moe-604m-a38m" in profile_lora.WORKLOAD_CHOICES
    legacy_names = {"moe" + "_3b", "qwen3" + "_30b" + "_a3b"}
    assert not legacy_names.intersection(profile_lora.WORKLOAD_CHOICES)
    assert "Qwen/Qwen3-30B-A3B" not in profile_lora.WORKLOAD_CHOICES
    assert profile_lora.KT_MOE_WORKLOADS == ("moe", "moe-604m-a75m", "moe-604m-a38m")
    assert profile_lora.is_hf_model_workload("Qwen/Qwen3-30B-A3B")
    assert profile_lora.is_moe_workload_name("Qwen/Qwen3-30B-A3B")


def test_profile_lora_e2e_driver_supports_workload_layer_specs() -> None:
    driver = _load_profile_lora_e2e_driver_module()

    specs = driver._expand_workloads(["moe-604m-a75m|2", "qwen|4", "Qwen/Qwen3-30B-A3B|1", "Qwen/Qwen3-30B-A3B|all"])

    assert [(spec.name, spec.profile_layers, spec.label) for spec in specs] == [
        ("moe-604m-a75m", 2, "moe-604m-a75m-l2"),
        ("dense_14b", 4, "dense_14b-l4"),
        ("Qwen/Qwen3-30B-A3B", 4, "qwen3-30b-a3b-l4"),
        ("Qwen/Qwen3-30B-A3B", 1, "qwen3-30b-a3b-l1"),
        ("Qwen/Qwen3-30B-A3B", "all", "qwen3-30b-a3b-lall"),
    ]


def test_profile_lora_e2e_driver_result_stem_includes_input_shape_and_recompute() -> None:
    driver = _load_profile_lora_e2e_driver_module()

    args = argparse.Namespace(mode="auto", precision="bf16", batch_size=16, seq_len=2048, activation_recompute=True)
    assert driver._result_stem(args, "asym", "nsys") == "bf16_lora-sft_b16_s2048_recomp_asym_nsys"

    args.activation_recompute = False
    assert driver._result_stem(args, "torch", "source") == "bf16_lora-sft_b16_s2048_norecomp_torch_source"

    args.expert_recompute_policy_spec = "tok-le64"
    assert driver._result_stem(args, "asym", "source") == "bf16_lora-sft_b16_s2048_norecomp_expertpolicytok-le64_asym_source"


def test_profile_lora_e2e_driver_expands_expert_recompute_policies() -> None:
    driver = _load_profile_lora_e2e_driver_module()

    args = argparse.Namespace(
        expert_recompute_policy=None,
        expert_recompute_policies=["none,tok-le16", "tok-ge32", "tok-le16"],
    )

    assert [spec.label for spec in driver._expert_recompute_policies(args)] == ["none", "tok-le16", "tok-ge32"]


def test_profile_lora_e2e_driver_parses_expert_recompute_policies() -> None:
    driver = _load_profile_lora_e2e_driver_module()

    specs = [
        driver.parse_expert_recompute_policy_spec(value)
        for value in ("none", "tok-le0", "tok-le0-act", "tok-le128", "tok-ge128", "tok64-256", "tok-ge128-act")
    ]

    assert [
        (
            spec.label,
            spec.policy,
            spec.token_threshold,
            spec.token_min,
            spec.token_max,
            spec.activation_save_policy,
            spec.activation_save_threshold,
            spec.activation_save_min,
            spec.activation_save_max,
            spec.force_custom_autograd,
        )
        for spec in specs
    ] == [
        ("none", "none", 0, 1, None, "save_all", 0, 1, None, False),
        ("tok-le0", "none", 0, 1, None, "save_all", 0, 1, None, True),
        ("tok-le0-act", "none", 0, 1, None, "save_all", 0, 1, None, True),
        ("tok-le128", "tok", 128, 1, 128, "save_all", 0, 1, None, False),
        ("tok-ge128", "tok", 0, 128, None, "save_all", 0, 1, None, False),
        ("tok64-256", "tok", 256, 64, 256, "save_all", 0, 1, None, False),
        ("tok-ge128-act", "none", 0, 1, None, "tok_act", 0, 128, None, False),
    ]


def test_profile_lora_e2e_driver_layer_recompute_only_applies_at_zero_expert_threshold() -> None:
    driver = _load_profile_lora_e2e_driver_module()

    assert driver._effective_activation_recompute(True, driver.parse_expert_recompute_policy_spec("none"))
    assert not driver._effective_activation_recompute(True, driver.parse_expert_recompute_policy_spec("tok-le0"))
    assert not driver._effective_activation_recompute(True, driver.parse_expert_recompute_policy_spec("tok-ge16"))


def test_activation_sweep_plot_rejects_legacy_expert_threshold_result_dirs() -> None:
    plotter = _load_activation_sweep_plot_module()

    meta = plotter.parse_result_dir(
        Path("/tmp/profiling/moe-604m-a38m-l2/bf16_lora-sft_b8_s256_norecomp_expertthr64_asym_nsys")
    )

    assert meta is None


def test_activation_sweep_plot_parses_expert_policy_flat_dirs() -> None:
    plotter = _load_activation_sweep_plot_module()

    ge_meta = plotter.parse_result_dir(
        Path("/tmp/profiling/lora__e2e__bf16/moe-604m-a38m-l2__b8_s1024_r64_a128/asym__nsys__norecomp__poltok-ge128/s1024")
    )
    bounded_meta = plotter.parse_result_dir(
        Path("/tmp/profiling/lora__e2e__bf16/moe-604m-a38m-l2__b8_s1024_r64_a128/asym__nsys__norecomp__poltok64-256/s1024")
    )

    assert ge_meta is not None
    assert ge_meta["expert_recompute_policy_spec"] == "tok-ge128"
    assert ge_meta["expert_recompute_policy"] == "tok"
    assert ge_meta["expert_recompute_threshold"] == 0
    assert ge_meta["expert_recompute_token_min"] == 128
    assert ge_meta["expert_recompute_token_max"] is None
    assert bounded_meta is not None
    assert bounded_meta["expert_recompute_policy_spec"] == "tok64-256"
    assert bounded_meta["expert_recompute_policy"] == "tok"
    assert bounded_meta["expert_recompute_threshold"] == 256
    assert bounded_meta["expert_recompute_token_min"] == 64
    assert bounded_meta["expert_recompute_token_max"] == 256


def test_profile_lora_e2e_driver_does_not_create_skipped_result_dirs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROFILE_LORA_E2E_DRIVER_PATH),
            "--workloads",
            "mlp",
            "--backends",
            "kt",
            "--profilers",
            "source",
            "--output-root",
            str(tmp_path),
            "--skip-summary",
        ],
        cwd=ROOT,
        check=True,
    )

    assert not list(tmp_path.rglob("*_kt_source"))


def test_profile_lora_e2e_driver_summary_row_reads_nested_nsys_profile() -> None:
    driver = _load_profile_lora_e2e_driver_module()
    row = driver._summary_row(
        workload="moe-604m-a38m-l1",
        backend="asym",
        profiler="nsys",
        device="cuda:0",
        physical_cuda_device=None,
        output_dir=ROOT / "profiling" / "moe-604m-a38m-l1",
        returncode=0,
        profile={
            "source_profile": {
                "step": {"total_milliseconds": 12.5},
                "memory": {
                    "gpu": {"peak_hbm_bytes": 4 * 1024 * 1024},
                    "cpu": {
                        "host_w_bytes": 2 * 1024 * 1024,
                        "pinned_total_bytes": 5 * 1024 * 1024,
                    },
                },
            }
        },
    )

    assert row["step_ms"] == pytest.approx(12.5)
    assert row["peak_hbm_bytes"] == 4 * 1024 * 1024
    assert row["expected_hbm_saved_bytes"] == 2 * 1024 * 1024
    assert row["hbm_saved_percent"] == pytest.approx(100.0 / 3.0)
    assert row["pinned_cpu_bytes"] == 5 * 1024 * 1024


def test_profile_lora_e2e_driver_prefers_nsys_stage_timing_over_source_step() -> None:
    driver = _load_profile_lora_e2e_driver_module()

    assert driver._profile_step_milliseconds(
        {
            "stages": [
                {"stage": "step.forward", "total_milliseconds": 7.0},
                {"stage": "step.backward", "total_milliseconds": 11.0},
            ],
            "source_profile": {"step": {"total_milliseconds": 99.0}},
        }
    ) == pytest.approx(18.0)


def test_profile_lora_e2e_driver_writes_descending_latency_and_memory_rankings() -> None:
    driver = _load_profile_lora_e2e_driver_module()
    summary = {
        "precision": "bf16",
        "workflow": "lora-sft",
        "runs": [
            {
                "workload": "fast",
                "backend": "asym",
                "profiler": "source",
                "status": "ok",
                "device": "cuda:0",
                "step_ms": 1.0,
                "peak_hbm_bytes": 30,
                "expected_hbm_saved_bytes": 0,
                "pinned_cpu_bytes": 0,
                "output_dir": "fast",
            },
            {
                "workload": "slow",
                "backend": "torch",
                "profiler": "source",
                "status": "ok",
                "device": "cuda:0",
                "step_ms": 3.0,
                "peak_hbm_bytes": 10,
                "expected_hbm_saved_bytes": 0,
                "pinned_cpu_bytes": 0,
                "output_dir": "slow",
            },
            {
                "workload": "memory-heavy",
                "backend": "asym",
                "profiler": "nsys",
                "status": "ok",
                "device": "cuda:0",
                "step_ms": 2.0,
                "peak_hbm_bytes": 50,
                "expected_hbm_saved_bytes": 0,
                "pinned_cpu_bytes": 0,
                "output_dir": "memory-heavy",
            },
        ],
    }

    latency_rows = [line for line in driver._latency_markdown(summary).splitlines() if line.startswith("| ") and not line.startswith("| Rank") and not line.startswith("|---")]
    memory_rows = [line for line in driver._memory_markdown(summary).splitlines() if line.startswith("| ") and not line.startswith("| Rank") and not line.startswith("|---")]

    assert [row.split("|")[2].strip() for row in latency_rows] == ["slow", "memory-heavy", "fast"]
    assert [row.split("|")[2].strip() for row in memory_rows] == ["memory-heavy", "fast", "slow"]
    assert "GPU HBM saved %" in driver._memory_markdown(summary)


def test_profile_lora_e2e_saved_tensor_buckets_keep_semantic_leaf_owners() -> None:
    profile_lora = _load_profile_lora_e2e_module()

    assert profile_lora._saved_tensor_bucket("forward.layers.0.mlp.silu_mul_activation") == "mlp.silu_mul_activation"
    assert profile_lora._saved_tensor_bucket("forward.layers.0.attention.sdpa") == "attention.sdpa"
    assert profile_lora._saved_tensor_bucket("forward.layers.0.attention.q_proj.lora_A") == "attention.q_proj.lora_A"
    assert profile_lora._saved_tensor_bucket("forward.layers.0.routed_expert.gate_lora") == "routed_expert.gate.lora"
    assert profile_lora._saved_tensor_bucket("forward.layers.0.routed_expert.gate_up_lora") == "routed_expert.gate_up.lora"


def test_postprocess_semantic_keys_keep_torch_base_and_gate_up_lora() -> None:
    postprocess = _load_postprocess_nsys_lora_module()

    assert postprocess._semantic_leaf_key("forward.layers.0.attention.q_proj_base") == "attention.q_proj.base_torch"
    assert postprocess._semantic_leaf_key("attention.q_proj.base_asymgemm") == "attention.q_proj.base_torch"
    assert postprocess._semantic_leaf_label("attention.q_proj.base_torch") == "Attention q_proj base torch"
    assert postprocess._semantic_leaf_key("forward.layers.0.routed_expert.gate_up_lora") == "routed_expert.gate_up.lora"


def test_postprocess_front_summary_uses_semantic_leaf_rows() -> None:
    postprocess = _load_postprocess_nsys_lora_module()
    report = {
        "stages": [
                {
                    "stage": "step.forward",
                    "total_milliseconds": 10.0,
                    "stage_breakdown": {"rows": [{"name": "cuda_memcpy_union", "milliseconds": 0.0, "percent": 0.0}]},
                "operation_kernel_classes": {
                    "rows": [
                        {
                            "operation": "forward.layers.0.mlp.silu_mul_activation",
                            "kernel_class": "Torch elementwise kernel",
                            "milliseconds": 0.7,
                        },
                        {
                            "operation": "forward.layers.0.attention.softmax",
                            "kernel_class": "Torch softmax kernel",
                            "milliseconds": 0.2,
                        },
                    ]
                },
                "gpu_no_kernel_gap_attribution": {
                    "rows": [
                        {
                            "name": "no-kernel host/autograd/Python: attention softmax",
                            "milliseconds": 0.05,
                        }
                    ]
                },
            },
            {
                "stage": "step.backward",
                "total_milliseconds": 20.0,
                "stage_breakdown": {"rows": [{"name": "cuda_memcpy_union", "milliseconds": 0.0}]},
                "operation_kernel_classes": {
                    "rows": [
                        {
                            "operation": "backward.layers.0.mlp.silu_mul_activation",
                            "kernel_class": "Torch elementwise kernel",
                            "milliseconds": 0.8,
                        },
                        {
                            "operation": "backward.layers.0.attention.softmax",
                            "kernel_class": "Torch softmax kernel",
                            "milliseconds": 0.3,
                        },
                    ]
                },
                "gpu_no_kernel_gap_attribution": {"rows": []},
            },
        ],
        "memory_profile": {
            "memory_attribution": {
                "saved_activations": {
                    "rows": [
                        {"owner": "mlp.silu_mul_activation", "unique_bytes": 1024 * 1024},
                        {"owner": "attention.softmax", "unique_bytes": 2 * 1024 * 1024},
                    ]
                }
            }
        },
    }

    rows = {row["key"]: row for row in postprocess._semantic_timing_memory_rows(report)}

    assert rows["mlp.silu_mul_activation"]["forward_gpu_ms"] == pytest.approx(0.7)
    assert rows["mlp.silu_mul_activation"]["backward_gpu_ms"] == pytest.approx(0.8)
    assert rows["mlp.silu_mul_activation"]["saved_activation_mib"] == pytest.approx(1.0)
    assert rows["mlp.silu_mul_activation"]["saved_activation_percent"] == pytest.approx(100.0 / 3.0)
    assert rows["attention.softmax"]["forward_gpu_ms"] == pytest.approx(0.2)
    assert rows["attention.softmax"]["forward_gap_ms"] == pytest.approx(0.05)
    assert rows["attention.softmax"]["saved_activation_mib"] == pytest.approx(2.0)
    assert rows["attention.softmax"]["saved_activation_percent"] == pytest.approx(200.0 / 3.0)


def test_postprocess_no_kernel_labels_are_canonical() -> None:
    postprocess = _load_postprocess_nsys_lora_module()

    assert postprocess._gap_semantic_key("No-kernel misc small gaps", "step.forward")[1] == "No-kernel forward misc small gaps"
    assert postprocess._gap_semantic_key("no-kernel misc small gaps", "step.backward")[1] == "No-kernel backward misc small gaps"
    assert (
        postprocess._gap_semantic_key("No-kernel host/autograd/Python: routed MoE / grouped", "step.forward")[1]
        == "No-kernel forward routed MoE / grouped"
    )
    assert (
        postprocess._gap_semantic_key(
            "No-kernel host/autograd/Python: forward unlabeled kernel chain (Torch copy/cast kernel -> Torch/CUTLASS GEMM kernel)",
            "step.forward",
        )[1]
        == "No-kernel forward unlabeled kernel chain (Torch copy/cast kernel -> Torch/CUTLASS GEMM kernel)"
    )


def test_postprocess_writes_separate_latency_and_memory_markdown() -> None:
    postprocess = _load_postprocess_nsys_lora_module()
    mib = 1024 * 1024
    empty_stage_details = {
        "host_api_breakdown": {"rows": []},
        "operation_kernel_time": {"rows": []},
        "operation_cuda_api_time": {"rows": []},
        "gpu_no_kernel_gap_attribution": {"rows": []},
        "gpu_no_kernel_gaps": {"rows": []},
    }
    report = {
        "source": "trace.sqlite",
        "stages": [
            {
                "stage": "step.forward",
                "total_milliseconds": 10.0,
                "stage_breakdown": {"rows": [{"name": "cuda_memcpy_union", "milliseconds": 0.0, "percent": 0.0}]},
                "operation_kernel_classes": {
                    "rows": [
                            {
                                "operation": "forward.layers.0.attention.softmax",
                                "kernel_class": "Torch softmax kernel",
                                "milliseconds": 2.0,
                                "percent": 20.0,
                            }
                    ]
                },
                **empty_stage_details,
            },
            {
                "stage": "step.backward",
                "total_milliseconds": 10.0,
                "stage_breakdown": {"rows": [{"name": "cuda_memcpy_union", "milliseconds": 0.0, "percent": 0.0}]},
                "operation_kernel_classes": {"rows": []},
                **empty_stage_details,
            },
        ],
        "source_profile": {
            "memory": {
                "gpu": {"peak_hbm_bytes": 4 * mib, "parameter_bytes": 1 * mib, "buffer_bytes": 0, "unattributed_peak_bytes": 3 * mib},
                "cpu": {"pinned_total_bytes": 2 * mib},
            },
            "stage_memory": {"rows": []},
        },
        "memory_profile": {
            "memory_attribution": {
                "categories": {"rows": [{"category": "gpu saved", "memory_space": "GPU HBM", "bytes": 3 * mib, "accuracy": "exact"}]},
                "saved_activations": {
                    "rows": [
                        {"owner": "attention.softmax", "unique_bytes": 5 * mib, "reference_bytes": 5 * mib, "save_count": 1, "unique_tensor_count": 1},
                    ],
                    "total_unique_bytes": 5 * mib,
                },
            }
        },
    }

    latency_text = postprocess.latency_markdown(report)
    memory_text = postprocess.memory_markdown(report)

    assert "## Top Latency" in latency_text
    assert "### Operation Kernel Classes" in latency_text
    assert "## Top Memory" not in latency_text
    assert "## Top Memory" in memory_text
    assert "## Fine-Grained Memory Attribution" in memory_text
    assert "### Operation Kernel Classes" not in memory_text
    assert memory_text.index("| memory attribution pass saved activation | attention.softmax | GPU HBM | 5242880 | 5.00 |") < memory_text.index("| source timing pass | GPU peak HBM | GPU HBM | 4194304 | 4.00 |")


def test_postprocess_memory_attribution_percentages_are_gpu_only() -> None:
    postprocess = _load_postprocess_nsys_lora_module()
    mib = 1024 * 1024
    profile = {
        "memory": {"gpu": {"peak_hbm_bytes": 4 * mib}},
        "memory_attribution": {
            "categories": {
                "rows": [
                    {"category": "cpu offload", "memory_space": "CPU pinned", "bytes": 20 * mib, "accuracy": "exact"},
                    {"category": "gpu saved", "memory_space": "GPU HBM", "bytes": 3 * mib, "accuracy": "exact"},
                    {"category": "gpu params", "memory_space": "GPU HBM", "bytes": 1 * mib, "accuracy": "exact"},
                ]
            },
            "saved_activations": {
                "rows": [
                    {"owner": "attention.softmax", "unique_bytes": 2 * mib, "reference_bytes": 4 * mib, "save_count": 2, "unique_tensor_count": 1},
                    {"owner": "mlp.silu_mul_activation", "unique_bytes": 1 * mib, "reference_bytes": 1 * mib, "save_count": 1, "unique_tensor_count": 1},
                ],
                "total_unique_bytes": 3 * mib,
            },
        }
    }

    text = "\n".join(postprocess._memory_attribution_markdown(profile))

    assert "Percent denominator: memory-attribution pass peak HBM `4194304` bytes." in text
    assert "| Category | Component | Memory space | bytes | MiB | % peak HBM | Accuracy |" in text
    assert "| cpu offload | - | CPU pinned | 20971520 | 20.00 | - | exact |" in text
    assert "| gpu saved | - | GPU HBM | 3145728 | 3.00 | 75.00% | exact |" in text
    assert "| gpu params | - | GPU HBM | 1048576 | 1.00 | 25.00% | exact |" in text
    assert "| Owner | unique bytes | unique MiB | % saved GPU | reference bytes | reference MiB | saves | unique tensors |" in text
    assert "| attention.softmax | 2097152 | 2.00 | 66.67% | 4194304 | 4.00 | 2 | 1 |" in text
    assert "| mlp.silu_mul_activation | 1048576 | 1.00 | 33.33% | 1048576 | 1.00 | 1 | 1 |" in text


def test_kt_backend_is_restricted_to_moe_workloads() -> None:
    profile_lora = _load_profile_lora_e2e_module()

    with pytest.raises(ValueError, match="backend=kt is only implemented for MoE LoRA SFT workloads"):
        profile_lora.validate_backend_workload(argparse.Namespace(backend="kt", workload="mlp", lora_dtype="bf16"))

    profile_lora.validate_backend_workload(argparse.Namespace(backend="kt", workload="moe", lora_dtype="bf16"))


def test_kt_backend_rejects_non_bf16_lora_dtype() -> None:
    profile_lora = _load_profile_lora_e2e_module()

    with pytest.raises(ValueError, match="backend=kt currently supports BF16 LoRA buffers only"):
        profile_lora.validate_backend_workload(argparse.Namespace(backend="kt", workload="moe", lora_dtype="fp16"))


def test_profile_lora_e2e_cli_rejects_kt_for_non_moe_workload(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROFILE_LORA_E2E_PATH),
            "--workload",
            "mlp",
            "--backend",
            "kt",
            "--device",
            "cpu",
            "--warmup-steps",
            "0",
            "--measure-steps",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "backend=kt is only implemented for MoE LoRA SFT workloads" in result.stderr


def test_kt_moe_wrapper_import_reports_clear_availability() -> None:
    try:
        wrapper_cls = _import_kt_moe_wrapper()
    except KTBackendUnavailable as exc:
        assert "backend=kt requires kt-kernel with SFT AMX support" in str(exc)
    else:
        assert wrapper_cls.__name__ == "KTMoEWrapper"
