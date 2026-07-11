from __future__ import annotations

import pytest
import torch

import asym_gemm.training.dense as dense_llm

TARGET_MODES = dense_llm.TARGET_MODES
DenseLLMConfig = dense_llm.DenseLLMConfig
MICRO_DENSE_LLM_CONFIG = dense_llm.MICRO_DENSE_LLM_CONFIG
SHOWCASE_DENSE_LLM_CONFIG = dense_llm.SHOWCASE_DENSE_LLM_CONFIG
adapter_state_names = dense_llm.adapter_state_names
build_model_pair = dense_llm.build_model_pair
estimate_dense_llm_parameters = dense_llm.estimate_dense_llm_parameters
run_adapter_reload_case = dense_llm.run_adapter_reload_case
run_memory_comparison = dense_llm.run_memory_comparison
run_parity_case = dense_llm.run_parity_case
run_repeated_steps = dense_llm.run_repeated_steps


def _direct_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] in {9, 10}


def _placement_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_on_device(tensor: torch.Tensor, device: torch.device) -> bool:
    if tensor.device.type != device.type:
        return False
    return device.index is None or tensor.device.index == device.index


def _fmt_mib(value: int | float) -> str:
    return f"{float(value) / (1024 ** 2):.3f} MiB"


def _print_parity(case: dict) -> None:
    print(
        "\n[M3 numerical error] "
        f"target_mode={case['target_mode']}, checkpointing={case['checkpointing']}, "
        f"logits_max_abs={case['logits_max_abs']:.6g}, "
        f"loss_abs={case['loss_abs']:.6g}, "
        f"input_grad_max_abs={case['input_grad_max_abs']:.6g}, "
        f"lora_grad_worst_max_abs={case['lora_grad_worst_max_abs']:.6g}, "
        f"activation_worst_max_abs={case['activation_worst_max_abs']:.6g}"
    )
    stats = case["execution_stats"]
    print(
        "[M3 execution] "
        f"asym_forward_calls={stats['asym_forward_calls']}, "
        f"asym_dx_calls={stats['asym_dx_calls']}, "
        f"staged_calls={stats['staged_calls']}, torch_calls={stats['torch_calls']}, "
        f"pinned_cpu={_fmt_mib(case['pinned_cpu_bytes'])}, "
        f"expected_hbm_saved={_fmt_mib(case['expected_hbm_saved_bytes'])}, "
        f"parity_peak_hbm={_fmt_mib(case['parity_peak_hbm_bytes'])}"
    )


def test_dense_llm_showcase_config_is_few_hundred_million_params() -> None:
    config = DenseLLMConfig()
    counts = estimate_dense_llm_parameters(config, target_mode="all", dtype=torch.bfloat16)
    print(
        "\n[M3 showcase parameter count] "
        f"config={config}, "
        f"total={counts['total_model_elements']:,}, "
        f"trainable_lora={counts['trainable_lora_elements']:,}, "
        f"cpu_frozen_targets={counts['cpu_resident_target_elements']:,}, "
        f"expected_hbm_saved={_fmt_mib(counts['expected_hbm_saved_bytes'])}, "
        f"lora_fraction={counts['lora_trainable_fraction']:.4%}"
    )

    assert config == SHOWCASE_DENSE_LLM_CONFIG
    assert config.lora_rank == 128
    assert counts["total_model_elements"] == 357_098_496
    assert counts["cpu_resident_target_elements"] == 150_994_944
    assert counts["trainable_lora_elements"] == 29_884_416
    assert set(counts["offload_modules"].split(",")) == {"gate_proj", "up_proj", "down_proj"}
    assert 300_000_000 <= counts["total_model_elements"] < 500_000_000
    assert counts["lora_trainable_fraction"] < 0.10


@pytest.mark.skipif(not _direct_bf16_available(), reason="M3 BF16 direct-fetch parity requires SM90/SM100")
@pytest.mark.parametrize("target_mode", TARGET_MODES)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_dense_llm_parity_by_target_mode_and_checkpointing(target_mode: str, checkpointing: bool) -> None:
    torch.manual_seed(0)
    case = run_parity_case(
        target_mode=target_mode,
        checkpointing=checkpointing,
        backend="asym",
        device="cuda",
        dtype=torch.bfloat16,
        config=MICRO_DENSE_LLM_CONFIG,
    )
    _print_parity(case)

    assert case["logits_max_abs"] <= 0.5
    assert case["loss_abs"] <= 0.5
    assert case["input_grad_max_abs"] <= 0.5
    assert case["lora_grad_worst_max_abs"] <= 0.5
    assert case["activation_worst_max_abs"] <= 0.5
    assert case["adapter_names_match_reference"]
    assert case["base_weight_grads_absent"]
    assert case["lora_grads_finite"]
    assert case["input_grad_finite"]
    assert case["direct_fetch_forward_used"]
    assert case["direct_fetch_dx_used"]
    assert case["execution_stats"]["staged_calls"] == 0
    assert case["execution_stats"]["torch_calls"] == 0
    assert case["expected_hbm_saved_bytes"] > 0
    assert case["pinned_cpu_bytes"] >= case["expected_hbm_saved_bytes"]


@pytest.mark.skipif(not _direct_bf16_available(), reason="M3 repeated-step BF16 check requires SM90/SM100")
def test_dense_llm_repeated_steps_are_finite_and_track_torch() -> None:
    report = run_repeated_steps(
        target_mode="all",
        checkpointing=False,
        backend="asym",
        device="cuda",
        dtype=torch.bfloat16,
        steps=5,
        config=MICRO_DENSE_LLM_CONFIG,
    )
    print(
        "\n[M3 repeated steps] "
        f"loss_curve={report['loss_curve']}, "
        f"reference_loss_curve={report['reference_loss_curve']}, "
        f"max_loss_relative_error={report['max_loss_relative_error']:.6g}, "
        f"optimizer={report['optimizer_state']['optimizer_class']}, "
        f"adam_state_bytes={_fmt_mib(report['optimizer_state']['state_tensor_bytes'])}, "
        f"pinned_cpu={_fmt_mib(report['pinned_cpu_bytes'])}, "
        f"expected_hbm_saved={_fmt_mib(report['expected_hbm_saved_bytes'])}"
    )

    assert report["all_finite"]
    assert report["max_loss_relative_error"] <= 0.02
    assert report["optimizer_contains_only_lora"]
    optimizer_state = report["optimizer_state"]
    assert optimizer_state["optimizer_class"] == "AdamW"
    assert optimizer_state["expected_kind"] == "lora"
    assert optimizer_state["all_expected_params_in_optimizer"]
    assert optimizer_state["only_expected_params_in_optimizer"]
    assert optimizer_state["state_for_all_expected_params"]
    assert optimizer_state["adam_moments_for_all_expected_params"]
    assert optimizer_state["state_entry_count"] == optimizer_state["expected_state_entry_count"]
    assert optimizer_state["unexpected_state_param_count"] == 0
    assert optimizer_state["unexpected_optimizer_param_count"] == 0
    assert optimizer_state["state_tensor_bytes"] > 0
    assert optimizer_state["non_finite_state_names"] == []
    assert report["frozen_base_unchanged"]
    assert report["base_weight_grads_absent"]
    assert report["direct_fetch_forward_used"]
    assert report["direct_fetch_dx_used"]
    assert report["execution_stats"]["staged_calls"] == 0
    assert report["execution_stats"]["torch_calls"] == 0


def test_dense_llm_adapter_state_names_and_reload_cpu() -> None:
    config = MICRO_DENSE_LLM_CONFIG
    asym_model, ref_model = build_model_pair(
        config=config,
        target_mode="all",
        backend="torch",
        device="cpu",
        dtype=torch.float32,
    )
    names = adapter_state_names(asym_model)
    expected = []
    for layer_idx in range(config.num_layers):
        for prefix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            expected.append(f"layers.{layer_idx}.{prefix}.lora_A.default.weight")
            expected.append(f"layers.{layer_idx}.{prefix}.lora_B.default.weight")

    assert names == expected
    assert adapter_state_names(ref_model) == expected

    reload_report = run_adapter_reload_case(
        target_mode="all",
        backend="torch",
        device="cpu",
        dtype=torch.float32,
        config=config,
    )
    print(
        "\n[M3 adapter state] "
        f"name_count={reload_report['adapter_state_name_count']}, "
        f"reload_changed_logits_max_abs={reload_report['reload_changed_logits_max_abs']:.6g}, "
        f"reload_matches_source_logits_max_abs={reload_report['reload_matches_source_logits_max_abs']:.6g}"
    )
    assert reload_report["adapter_state_names"] == expected
    assert reload_report["names_have_peft_shape"]
    assert reload_report["base_keys_absent"]
    assert reload_report["reload_changed_logits_max_abs"] > 0.0
    assert reload_report["reload_matches_source_logits_max_abs"] <= 1e-6


def test_dense_llm_lora_all_offload_mlp_placement() -> None:
    config = MICRO_DENSE_LLM_CONFIG
    device = _placement_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, _ = build_model_pair(
        config=config,
        target_mode="all",
        target_modules="all",
        offload_modules="mlp",
        backend="asym",
        device=device,
        dtype=dtype,
        seed=124,
        lora_seed=125,
    )

    assert model.target_names == dense_llm.ALL_TARGETS
    assert model.offload_names == dense_llm.MLP_TARGETS
    assert len(adapter_state_names(model)) == config.num_layers * len(dense_llm.ALL_TARGETS) * 2

    for layer in model.layers:
        attention_modules = (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        )
        mlp_modules = (
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        )
        for module in attention_modules:
            assert isinstance(module, dense_llm.TorchLoRALinear)
            assert _is_on_device(module.base_layer.weight, device)
            assert _is_on_device(module.lora_A["default"].weight, device)
            assert _is_on_device(module.lora_B["default"].weight, device)
            assert module.lora_A["default"].weight.dtype == torch.bfloat16
            assert module.lora_B["default"].weight.dtype == torch.bfloat16
        for module in mlp_modules:
            assert isinstance(module, dense_llm.AsymLoRALinear)
            assert module.base_layer.host_weight.weight.device.type == "cpu"
            assert not module.base_layer.host_weight.weight.requires_grad
            if device.type == "cuda":
                assert module.base_layer.host_weight.weight.is_pinned()
            assert _is_on_device(module.lora_A["default"].weight, device)
            assert _is_on_device(module.lora_B["default"].weight, device)
            assert module.lora_A["default"].weight.dtype == torch.bfloat16
            assert module.lora_B["default"].weight.dtype == torch.bfloat16

    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    expected_offloaded_mlp_bytes = config.num_layers * len(dense_llm.MLP_TARGETS) * config.intermediate_size * config.hidden_size * dtype_bytes
    expected_gpu_attention_bytes = config.num_layers * len(dense_llm.ATTENTION_TARGETS) * config.hidden_size * config.hidden_size * dtype_bytes
    assert model.cpu_resident_base_weight_bytes == expected_offloaded_mlp_bytes
    assert model.gpu_resident_target_weight_bytes == expected_gpu_attention_bytes


@pytest.mark.skipif(not _direct_bf16_available(), reason="CUDA SM90/SM100 required for M3 HBM direct-fetch accounting")
def test_dense_llm_cpu_resident_targets_save_hbm() -> None:
    memory = run_memory_comparison(
        target_mode="all",
        backend="asym",
        device="cuda",
        dtype=torch.bfloat16,
        config=MICRO_DENSE_LLM_CONFIG,
    )
    normal = memory["normal_gpu_resident"]
    asym = memory["asym_cpu_resident"]
    print(
        "\n[M3 memory comparison] "
        f"normal_model_hbm={_fmt_mib(normal['model_hbm_bytes'])}, "
        f"asym_model_hbm={_fmt_mib(asym['model_hbm_bytes'])}, "
        f"hbm_model_saved={_fmt_mib(memory['hbm_model_saved_bytes'])}, "
        f"normal_peak_hbm={_fmt_mib(normal['peak_hbm_bytes'])}, "
        f"asym_peak_hbm={_fmt_mib(asym['peak_hbm_bytes'])}, "
        f"pinned_cpu={_fmt_mib(memory['pinned_cpu_bytes'])}, "
        f"expected_hbm_saved={_fmt_mib(memory['expected_hbm_saved_bytes'])}"
    )

    assert asym["cpu_resident_base_weight_bytes"] == memory["expected_hbm_saved_bytes"]
    assert memory["expected_hbm_saved_bytes"] > 0
    assert memory["pinned_cpu_bytes"] >= memory["expected_hbm_saved_bytes"]
    assert normal["model_hbm_bytes"] > asym["model_hbm_bytes"]
    assert memory["hbm_model_saved_bytes"] >= int(memory["expected_hbm_saved_bytes"] * 0.8)
    optimizer_state = asym["optimizer_state"]
    assert optimizer_state["optimizer_class"] == "AdamW"
    assert optimizer_state["all_expected_params_in_optimizer"]
    assert optimizer_state["only_expected_params_in_optimizer"]
    assert optimizer_state["adam_moments_for_all_expected_params"]
    assert optimizer_state["state_entry_count"] == optimizer_state["expected_state_entry_count"]
    assert optimizer_state["unexpected_state_param_count"] == 0
    assert optimizer_state["unexpected_optimizer_param_count"] == 0
    assert optimizer_state["state_tensor_bytes"] > 0
    assert memory["direct_fetch_forward_used"]
    assert memory["direct_fetch_dx_used"]
    assert asym["execution_stats"]["staged_calls"] == 0
    assert asym["execution_stats"]["torch_calls"] == 0
