from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from asym_gemm.training import MICRO_DENSE_LLM_CONFIG, MICRO_MOE_CONFIG


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def _load_report(name: str) -> dict[str, Any]:
    path = REPORTS / name
    assert path.exists(), f"missing report: {path}"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_adamw_state(state: Mapping[str, Any], *, expected_kind: str) -> None:
    assert state["optimizer_class"] == "AdamW"
    assert state["expected_kind"] == expected_kind
    assert state["all_expected_params_in_optimizer"]
    assert state["only_expected_params_in_optimizer"]
    assert state["state_for_all_expected_params"]
    assert state["adam_moments_for_all_expected_params"]
    assert state["state_entry_count"] == state["expected_state_entry_count"]
    assert state["unexpected_state_param_count"] == 0
    assert state["unexpected_optimizer_param_count"] == 0
    assert state["state_tensor_bytes"] > 0
    assert state["non_finite_state_names"] == []


def _assert_transformer_moe_schema(report: Mapping[str, Any]) -> None:
    config = report["config"]
    for field in ("vocab_size", "num_heads", "hidden_size", "num_layers"):
        assert field in config, f"missing transformer config field in M4 report: {field}"
    assert "logical_tokens" in config or {"batch_size", "seq_len"} <= set(config)
    assert config["num_shared_experts"] > 0

    architecture = report.get("architecture") or report.get("model_architecture")
    if architecture is not None:
        assert isinstance(architecture, Mapping), "M4 report architecture must be a mapping when present"
        assert architecture["is_transformer_style"]
        assert architecture["has_token_embeddings"]
        assert architecture["has_attention"]
        assert architecture["has_layernorm"]
        assert architecture["has_lm_head"]
        assert architecture["shared_expert_count"] > 0
        assert architecture["routed_expert_count"] > 0

    accounting = report["parameter_accounting"]
    numeric_items = [(key, value) for key, value in accounting.items() if isinstance(value, int | float)]
    assert accounting["trainable_lora_elements"] > 0
    assert accounting["trainable_router_elements"] > 0
    assert any("attention" in key and value > 0 for key, value in numeric_items)
    assert any("embedding" in key and value > 0 for key, value in numeric_items)
    assert any("shared" in key and "expert" in key and value > 0 for key, value in numeric_items)
    assert any(("routed" in key or key == "frozen_expert_base_elements") and "expert" in key and value > 0 for key, value in numeric_items)


def test_m3_checked_in_report_matches_current_rank128_direct_adamw_schema() -> None:
    report = _load_report("m3_tiny_llm.json")
    assert report["status"] == "pass"
    assert report["config"]["lora_rank"] == MICRO_DENSE_LLM_CONFIG.lora_rank == 128
    assert report["showcase_parameter_accounting"]["total_model_elements"] == 357_098_496
    assert report["parameter_accounting"]["trainable_lora_elements"] == 1_114_112

    for case in report["parity_cases"]:
        assert case["direct_fetch_forward_used"]
        assert case["direct_fetch_dx_used"]
        assert case["execution_stats"]["staged_calls"] == 0
        assert case["execution_stats"]["torch_calls"] == 0
        assert case["lora_grad_worst_max_abs"] <= 0.5

    _assert_adamw_state(report["repeated_steps"]["optimizer_state"], expected_kind="lora")
    memory = report["memory_comparison"]
    assert memory["direct_fetch_forward_used"]
    assert memory["direct_fetch_dx_used"]
    assert memory["expected_hbm_saved_bytes"] > 0
    assert memory["hbm_model_saved_bytes"] > 0
    assert memory["asym_cpu_resident"]["execution_stats"]["staged_calls"] == 0
    assert memory["asym_cpu_resident"]["execution_stats"]["torch_calls"] == 0


def test_m4_checked_in_report_matches_current_rank128_direct_adamw_schema() -> None:
    report = _load_report("m4_tiny_moe.json")
    assert report["status"] == "pass"
    _assert_transformer_moe_schema(report)
    assert report["config"]["lora_rank"] == MICRO_MOE_CONFIG.lora_rank == 128
    assert report["parameter_accounting"]["trainable_elements"] > report["parameter_accounting"]["trainable_router_elements"]
    assert set(report["route_patterns_tested"]) == {"balanced", "empty", "skewed", "repeated"}
    assert report["metadata_modes_tested"] == ["contiguous", "masked"]

    patterns = set()
    modes = set()
    for case in report["parity"]:
        modes.add(case["mode"])
        assert case["stats"]["staged_calls"] == 0
        assert case["stats"]["torch_calls"] == 0
        assert case["stats"]["asym_forward_calls"] > 0
        assert case["stats"]["asym_dx_calls"] > 0
        assert case["output_max_abs"] <= 0.5
        assert case["loss_abs"] <= 0.05
        assert case["input_grad_max_abs"] <= 0.05
        assert case["lora_grad_worst_max_abs"] <= 0.05
        if case["learned_router"]:
            assert case["route_pattern"] == "learned"
            assert case["router_grad_worst_max_abs"] <= 0.05
        else:
            patterns.add(case["route_pattern"])
    assert modes == {"contiguous", "masked"}
    assert patterns == set(report["route_patterns_tested"])

    toy = report["toy_training"]
    assert toy["all_losses_finite"]
    assert toy["used_static_coverage_step"]
    assert toy["used_learned_router_step"]
    _assert_adamw_state(toy["optimizer_state"], expected_kind="lora_plus_router")

    frozen = toy["frozen_host_weight_summary"]
    assert frozen["host_weight_count"] > 0
    assert frozen["all_cpu"]
    assert frozen["all_requires_grad_false"]
    assert frozen["all_grads_absent"]
    assert frozen["all_unchanged"]
    assert frozen["absent_from_named_parameters"]
    assert frozen["absent_from_optimizer_params"]
    assert frozen["absent_from_optimizer_state"]

    memory = report["memory_comparison"]
    assert memory["direct_fetch_forward_used"]
    assert memory["direct_fetch_dx_used"]
    assert memory["expected_hbm_saved_bytes"] > 0
    assert memory["hbm_model_saved_bytes"] > 0
    assert memory["asym_cpu_resident"]["execution_stats"]["staged_calls"] == 0
    assert memory["asym_cpu_resident"]["execution_stats"]["torch_calls"] == 0
