from __future__ import annotations

import gc
import inspect
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from asym_gemm.training import AsymExecutionStats
import asym_gemm.training.moe as tiny_moe


TinyMoEConfig = tiny_moe.TinyMoEConfig
MICRO_MOE_CONFIG = tiny_moe.MICRO_MOE_CONFIG
SHOWCASE_MOE_CONFIG = tiny_moe.SHOWCASE_MOE_CONFIG
estimate_tiny_moe_parameters = tiny_moe.estimate_tiny_moe_parameters
GROUPED_MODES = ("contiguous", "masked")
STATIC_PATTERNS = ("balanced", "empty", "skewed", "repeated")


def _direct_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] in {9, 10}


def _fmt_mib(value: int | float) -> str:
    return f"{float(value) / (1024 ** 2):.3f} MiB"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _make_input(config: TinyMoEConfig, *, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(_route_token_count(config), config.hidden_size, generator=generator, dtype=torch.float32) * 0.5
    return x.to(device=device, dtype=dtype)


def _loss(y: torch.Tensor) -> torch.Tensor:
    return y.float().square().mean() + y.float()[:, :4].sum() * 0.0003


def _route_token_count(config: TinyMoEConfig) -> int:
    if hasattr(config, "logical_tokens"):
        return int(config.logical_tokens)
    return int(config.batch_size) * int(config.seq_len)


def _batch_size(config: TinyMoEConfig) -> int:
    return int(getattr(config, "batch_size", 1))


def _seq_len(config: TinyMoEConfig) -> int:
    return int(getattr(config, "seq_len", _route_token_count(config) // _batch_size(config)))


def _is_lora_name(name: str) -> bool:
    return "lora" in name.lower()


def _is_router_name(name: str) -> bool:
    return "router" in name.lower()


def _is_trainable_moe_name(name: str) -> bool:
    return _is_lora_name(name) or _is_router_name(name)


def _transformer_config_fields(config: TinyMoEConfig) -> tuple[str, ...]:
    return tuple(name for name in ("vocab_size", "batch_size", "seq_len", "num_heads", "head_dim") if hasattr(config, name))


def _assert_transformer_config(config: TinyMoEConfig) -> None:
    required = ("vocab_size", "num_heads")
    missing = [name for name in required if not hasattr(config, name)]
    assert not missing, f"TinyMoEConfig must mirror tiny_dense_llm transformer fields; missing={missing}"
    assert hasattr(config, "logical_tokens") or (hasattr(config, "batch_size") and hasattr(config, "seq_len"))
    assert int(config.vocab_size) > 0
    assert _batch_size(config) > 0
    assert _seq_len(config) > 1
    assert int(config.num_heads) > 0
    assert int(config.hidden_size) % int(config.num_heads) == 0


def _make_transformer_inputs(
    config: TinyMoEConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(
        _batch_size(config),
        _seq_len(config),
        int(config.hidden_size),
        generator=generator,
        dtype=torch.float32,
    ) * 0.03
    labels = torch.randint(
        low=0,
        high=int(config.vocab_size),
        size=(_batch_size(config), _seq_len(config)),
        generator=generator,
        dtype=torch.long,
    )
    return inputs.to(device=device, dtype=dtype), labels.to(device=device)


def _forward_accepts_transformer_inputs(model: torch.nn.Module) -> bool:
    parameters = inspect.signature(model.forward).parameters
    return "inputs_embeds" in parameters or "input_ids" in parameters


def _call_model(
    model: torch.nn.Module,
    *,
    config: TinyMoEConfig,
    inputs: torch.Tensor,
    labels: torch.Tensor | None,
    static_routing: Any,
    mode: str,
) -> Any:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {}
    if "labels" in parameters and labels is not None:
        kwargs["labels"] = labels
    if "static_routing" in parameters:
        kwargs["static_routing"] = static_routing
    if "mode" in parameters:
        kwargs["mode"] = mode
    if "return_activations" in parameters:
        kwargs["return_activations"] = True

    if "inputs_embeds" in parameters:
        return model(inputs_embeds=inputs, **kwargs)
    if "input_ids" in parameters:
        input_ids = torch.zeros(
            (_batch_size(config), _seq_len(config)),
            device=inputs.device,
            dtype=torch.long,
        )
        return model(input_ids=input_ids, **kwargs)
    return model(inputs, **kwargs)


def _output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "hidden_states", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"model output does not expose a tensor payload: {type(output).__name__}")


def _loss_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, dict) and isinstance(output.get("loss"), torch.Tensor):
        return output["loss"]
    y = _output_tensor(output)
    if y.dim() == 2:
        return _loss(y)
    return y.float().square().mean() + y.float().reshape(-1, y.shape[-1])[:, :4].sum() * 0.0003


def _max_abs_tensor(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    assert lhs is not None
    assert rhs is not None
    return tiny_moe.max_abs_error(lhs, rhs)


def _grad_worst_allow_missing(
    lhs: torch.nn.Module,
    rhs: torch.nn.Module,
    predicate,
) -> tuple[float, list[str]]:
    lhs_params = dict(lhs.named_parameters())
    rhs_params = dict(rhs.named_parameters())
    worst = 0.0
    compared = 0
    missing: list[str] = []
    for name, lhs_param in lhs_params.items():
        if not predicate(name):
            continue
        rhs_param = rhs_params[name]
        if lhs_param.grad is None or rhs_param.grad is None:
            missing.append(name)
            continue
        compared += 1
        worst = max(worst, tiny_moe.max_abs_error(lhs_param.grad, rhs_param.grad))
    if compared == 0:
        raise AssertionError("no active gradients found for requested parameter set")
    return worst, missing


def _host_weight_items(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    items: list[tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        host_weight = getattr(module, "host_weight", None)
        weight = getattr(host_weight, "weight", None)
        if isinstance(weight, torch.Tensor):
            items.append((name, weight))
    return items


def _assert_transformer_moe_modules(model: torch.nn.Module, config: TinyMoEConfig) -> None:
    _assert_transformer_config(config)
    assert _forward_accepts_transformer_inputs(model), "TinyMoE forward must accept token-style transformer inputs"

    module_names = [name for name, _ in model.named_modules()]
    all_names = module_names + [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
    layernorm_count = sum(
        1 for name, module in model.named_modules() if isinstance(module, torch.nn.LayerNorm) or "layernorm" in name.lower()
    )
    assert any("embed" in name for name in all_names), "expected token/position embedding state"
    assert any("lm_head" in name or name.endswith("head") for name in all_names), "expected LM head state"
    assert layernorm_count >= int(config.num_layers) * 2
    assert any("self_attn" in name or ".q_proj" in name or ".k_proj" in name for name in all_names), "expected attention projections"
    assert any(_is_router_name(name) for name in all_names), "expected learned router state"
    assert any("shared" in name.lower() and ("expert" in name.lower() or "mlp" in name.lower()) for name in all_names)
    assert any("expert" in name.lower() and "shared" not in name.lower() for name in all_names)


def _assert_transformer_forward_contract(model: torch.nn.Module, config: TinyMoEConfig, *, device: torch.device, dtype: torch.dtype) -> None:
    inputs, labels = _make_transformer_inputs(config, device=device, dtype=dtype, seed=219)
    inputs = inputs.detach().clone().requires_grad_(True)
    output = _call_model(model, config=config, inputs=inputs, labels=labels, static_routing=None, mode="contiguous")
    assert isinstance(output, dict), "transformer-style TinyMoE should return a dict with logits/loss"
    logits = output.get("logits")
    assert isinstance(logits, torch.Tensor)
    assert tuple(logits.shape[:2]) == (_batch_size(config), _seq_len(config))
    assert int(logits.shape[-1]) == int(config.vocab_size)
    loss = _loss_from_output(output)
    assert loss.dim() == 0
    loss.backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad.detach().float()).all().item())


def _assert_tiny_moe_report_contract(report: dict[str, Any]) -> None:
    config = report["config"]
    for field in ("vocab_size", "num_heads", "hidden_size", "num_layers"):
        assert field in config, f"missing transformer config field in report: {field}"
    assert "logical_tokens" in config or {"batch_size", "seq_len"} <= set(config)
    assert int(config.get("num_shared_experts", 0)) > 0

    accounting = report["parameter_accounting"]
    numeric_items = [(key, value) for key, value in accounting.items() if isinstance(value, int | float)]
    assert any("attention" in key and value > 0 for key, value in numeric_items)
    assert any("embedding" in key and value > 0 for key, value in numeric_items)
    assert any("shared" in key and "expert" in key and value > 0 for key, value in numeric_items)
    assert any(("routed" in key or key == "frozen_expert_base_elements") and "expert" in key and value > 0 for key, value in numeric_items)

    architecture = report.get("architecture") or report.get("model_architecture")
    if architecture is None:
        return
    assert isinstance(architecture, dict), "report architecture must be a mapping when present"
    assert architecture.get("is_transformer_style") is True
    assert architecture.get("has_attention") is True
    assert architecture.get("has_layernorm") is True
    assert architecture.get("has_lm_head") is True
    assert architecture.get("has_token_embeddings") is True
    assert int(architecture.get("shared_expert_count", 0)) > 0
    assert int(architecture.get("routed_expert_count", 0)) > 0


def test_tiny_moe_showcase_config_is_defensible_transformer_moe() -> None:
    config = TinyMoEConfig()
    counts = estimate_tiny_moe_parameters(config, dtype=torch.bfloat16)
    _assert_transformer_config(config)
    shared_keys = [key for key in counts if "shared" in key and "expert" in key and "elements" in key]
    routed_keys = [
        key
        for key in counts
        if ("routed" in key or key == "frozen_expert_base_elements") and "expert" in key and "elements" in key
    ]
    print(
        "\n[M4 showcase parameter count] "
        f"config={config}, "
        f"total={counts['total_model_elements']:,}, "
        f"trainable={counts['trainable_elements']:,}, "
        f"transformer_fields={_transformer_config_fields(config)}, "
        f"shared_keys={shared_keys}, routed_keys={routed_keys}, "
        f"expected_hbm_saved={_fmt_mib(counts['expected_hbm_saved_bytes'])}, "
        f"trainable_fraction={counts['trainable_fraction']:.4%}"
    )

    assert config == SHOWCASE_MOE_CONFIG
    assert config.lora_rank == 128
    assert any(int(counts[key]) > 0 for key in shared_keys), "accounting must include frozen shared experts"
    assert any(int(counts[key]) > 0 for key in routed_keys), "accounting must include frozen routed experts"
    assert any("attention" in key and int(counts[key]) > 0 for key in counts)
    assert any("embedding" in key and int(counts[key]) > 0 for key in counts)
    assert counts["expected_hbm_saved_bytes"] > 0
    assert counts["total_model_elements"] > 500_000_000
    assert counts["trainable_fraction"] < 0.20


def _lora_grad_worst_allow_missing(lhs: torch.nn.Module, rhs: torch.nn.Module) -> tuple[float, list[str]]:
    return _grad_worst_allow_missing(lhs, rhs, _is_lora_name)


def _router_grad_worst(lhs: torch.nn.Module, rhs: torch.nn.Module) -> float:
    worst, missing = _grad_worst_allow_missing(lhs, rhs, _is_router_name)
    if missing:
        raise AssertionError(f"missing router gradients: {missing}")
    return worst


def test_tiny_moe_micro_model_is_transformer_style_with_shared_and_routed_experts() -> None:
    config = MICRO_MOE_CONFIG
    device = torch.device("cpu")
    dtype = torch.float32
    asym, ref, _, _ = tiny_moe.make_tiny_moe_pair(
        config=config,
        seed=210,
        device=device,
        base_dtype=dtype,
        backend="torch_only",
        pin_memory=False,
    )

    _assert_transformer_moe_modules(asym, config)
    _assert_transformer_moe_modules(ref, config)
    _assert_transformer_forward_contract(asym, config, device=device, dtype=dtype)


def test_tiny_moe_asym_base_weights_are_grouped_host_stacks() -> None:
    config = MICRO_MOE_CONFIG
    device = torch.device("cpu")
    dtype = torch.float32
    asym, _, _, _ = tiny_moe.make_tiny_moe_pair(
        config=config,
        seed=211,
        device=device,
        base_dtype=dtype,
        backend="torch_only",
        pin_memory=False,
    )

    host_weights = _host_weight_items(asym)
    host_names = [name for name, _ in host_weights]
    expected_per_layer = 3 + (3 if config.num_shared_experts > 0 else 0)

    assert len(host_weights) == config.num_layers * expected_per_layer
    assert all(weight.dim() == 3 for _, weight in host_weights)
    assert any("expert_gate_base" in name for name in host_names)
    assert any("expert_up_base" in name for name in host_names)
    assert any("expert_down_base" in name for name in host_names)
    assert any("shared_gate_base" in name for name in host_names)
    assert all(".experts." not in name for name in host_names)
    assert all(weight.device.type == "cpu" and not weight.requires_grad for _, weight in host_weights)


def _run_static_parity(pattern: str, mode: str) -> dict[str, Any]:
    config = MICRO_MOE_CONFIG
    device = torch.device("cpu")
    dtype = torch.float32
    asym, ref, _, stats = tiny_moe.make_tiny_moe_pair(
        config=config,
        seed=300,
        device=device,
        base_dtype=dtype,
        backend="torch_only",
        pin_memory=False,
    )
    routes = tiny_moe.make_static_routes(config, device, pattern=pattern)
    labels = None
    if _forward_accepts_transformer_inputs(asym):
        x, labels = _make_transformer_inputs(config, device=device, dtype=dtype, seed=301)
    else:
        x = _make_input(config, device=device, dtype=dtype, seed=301)
    x = x.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = _call_model(asym, config=config, inputs=x, labels=labels, static_routing=routes, mode=mode)
    out_ref = _call_model(ref, config=config, inputs=x_ref, labels=labels, static_routing=routes, mode=mode)
    y = _output_tensor(out)
    y_ref = _output_tensor(out_ref)
    loss = _loss_from_output(out)
    loss_ref = _loss_from_output(out_ref)
    loss.backward()
    loss_ref.backward()

    metadata = tiny_moe.build_route_metadata(
        routes[0][0],
        routes[0][1],
        num_experts=config.num_experts,
        mode=mode,
    )
    summary = tiny_moe.route_metadata_summary(metadata)
    repeated_pairs = int((routes[0][0][:, 0] == routes[0][0][:, 1]).sum().item())
    lora_worst, missing_lora_grads = _lora_grad_worst_allow_missing(asym, ref)
    return {
        "pattern": pattern,
        "mode": mode,
        "output_max_abs": tiny_moe.max_abs_error(y, y_ref),
        "loss_abs": abs(float(loss.item()) - float(loss_ref.item())),
        "input_grad_max_abs": _max_abs_tensor(x.grad, x_ref.grad),
        "lora_grad_worst_max_abs": lora_worst,
        "missing_lora_grad_names": missing_lora_grads,
        "route_metadata": summary,
        "repeated_token_expert_pairs": repeated_pairs,
        "execution_stats": stats.as_dict(),
    }


def _run_learned_router_parity(mode: str) -> dict[str, Any]:
    config = MICRO_MOE_CONFIG
    device = torch.device("cpu")
    dtype = torch.float32
    asym, ref, _, stats = tiny_moe.make_tiny_moe_pair(
        config=config,
        seed=310,
        device=device,
        base_dtype=dtype,
        backend="torch_only",
        pin_memory=False,
    )
    labels = None
    if _forward_accepts_transformer_inputs(asym):
        x, labels = _make_transformer_inputs(config, device=device, dtype=dtype, seed=311)
    else:
        x = _make_input(config, device=device, dtype=dtype, seed=311)
    x = x.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = _call_model(asym, config=config, inputs=x, labels=labels, static_routing=None, mode=mode)
    out_ref = _call_model(ref, config=config, inputs=x_ref, labels=labels, static_routing=None, mode=mode)
    y = _output_tensor(out)
    y_ref = _output_tensor(out_ref)
    loss = _loss_from_output(out)
    loss_ref = _loss_from_output(out_ref)
    loss.backward()
    loss_ref.backward()
    return {
        "mode": mode,
        "output_max_abs": tiny_moe.max_abs_error(y, y_ref),
        "loss_abs": abs(float(loss.item()) - float(loss_ref.item())),
        "input_grad_max_abs": _max_abs_tensor(x.grad, x_ref.grad),
        "lora_grad_worst_max_abs": _lora_grad_worst_allow_missing(asym, ref)[0],
        "router_grad_worst_max_abs": _router_grad_worst(asym, ref),
        "execution_stats": stats.as_dict(),
    }


def _memory_probe(
    *,
    model_kind: str,
    state: dict[str, Any],
    config: TinyMoEConfig,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    _clear_cuda(device)
    hbm_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    stats = AsymExecutionStats()
    if model_kind == "normal_gpu_resident":
        model = tiny_moe.TorchTinyMoEReference(state, config=config, device=device, base_dtype=dtype)
    elif model_kind == "asym_cpu_resident":
        model = tiny_moe.TinyMoE(
            state,
            config=config,
            device=device,
            base_dtype=dtype,
            backend=backend,
            pin_memory=device.type == "cuda",
            stats=stats,
        )
    else:
        raise ValueError(f"unknown model_kind={model_kind!r}")
    _sync(device)
    model_hbm = int(torch.cuda.memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    x = _make_input(config, device=device, dtype=dtype, seed=401).requires_grad_(True)
    y = model(x, mode="contiguous")
    assert isinstance(y, torch.Tensor)
    loss = _loss(y)
    loss.backward()
    _sync(device)
    peak_hbm = int(torch.cuda.max_memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    result = {
        "mode": model_kind,
        "model_hbm_bytes": max(0, model_hbm),
        "peak_hbm_bytes": max(0, peak_hbm),
        "frozen_weight_bytes": int(model.frozen_weight_bytes),
        "pinned_cpu_bytes": int(getattr(model, "pinned_cpu_bytes", 0)),
        "execution_stats": stats.as_dict(),
    }
    del y, loss, x, model
    _clear_cuda(device)
    return result


def _actual_memory_comparison(*, backend: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    config = MICRO_MOE_CONFIG
    state = tiny_moe.make_tiny_moe_state(config, seed=400, base_dtype=dtype)
    normal = _memory_probe(
        model_kind="normal_gpu_resident",
        state=state,
        config=config,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    asym = _memory_probe(
        model_kind="asym_cpu_resident",
        state=state,
        config=config,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    return {
        "normal_gpu_resident": normal,
        "asym_cpu_resident": asym,
        "hbm_model_saved_bytes": normal["model_hbm_bytes"] - asym["model_hbm_bytes"],
        "hbm_peak_saved_bytes": normal["peak_hbm_bytes"] - asym["peak_hbm_bytes"],
        "expected_hbm_saved_bytes": asym["frozen_weight_bytes"],
        "pinned_cpu_bytes": asym["pinned_cpu_bytes"],
    }


def _print_static(case: dict[str, Any]) -> None:
    print(
        "\n[M4 static routing] "
        f"mode={case['mode']}, pattern={case['pattern']}, "
        f"output_max_abs={case['output_max_abs']:.6g}, "
        f"loss_abs={case['loss_abs']:.6g}, "
        f"input_grad_max_abs={case['input_grad_max_abs']:.6g}, "
        f"lora_grad_worst_max_abs={case['lora_grad_worst_max_abs']:.6g}, "
        f"metadata={case['route_metadata']}, "
        f"fallbacks={case['execution_stats']}"
    )


def _print_learned(case: dict[str, Any]) -> None:
    print(
        "\n[M4 learned router] "
        f"mode={case['mode']}, output_max_abs={case['output_max_abs']:.6g}, "
        f"loss_abs={case['loss_abs']:.6g}, "
        f"input_grad_max_abs={case['input_grad_max_abs']:.6g}, "
        f"lora_grad_worst_max_abs={case['lora_grad_worst_max_abs']:.6g}, "
        f"router_grad_worst_max_abs={case['router_grad_worst_max_abs']:.6g}, "
        f"fallbacks={case['execution_stats']}"
    )


def _print_report(report: dict[str, Any]) -> None:
    memory = report["memory"]
    print(
        "\n[M4 report] "
        f"backend={report['backend']}, dtype={report['base_dtype']}, "
        f"direct_forward={report['direct_fetch_forward_used']}, "
        f"direct_dx={report['direct_fetch_dx_used']}, "
        f"fallbacks={report['fallback_counts']}, "
        f"toy_steps={report['toy_training']['steps']}, "
        f"optimizer={report['toy_training']['optimizer_state']['optimizer_class']}, "
        f"adam_state={_fmt_mib(report['toy_training']['optimizer_state']['state_tensor_bytes'])}, "
        f"toy_seconds_per_step={report['toy_training']['seconds_per_step']:.6g}"
    )
    print(
        "[M4 report memory] "
        f"peak_hbm={_fmt_mib(memory['peak_hbm_bytes'])}, "
        f"model_hbm_saved={_fmt_mib(memory['hbm_model_saved_bytes'])}, "
        f"expected_hbm_saved={_fmt_mib(memory['expected_hbm_saved_bytes'])}, "
        f"pinned_cpu={_fmt_mib(memory['pinned_cpu_bytes'])}"
    )


def _print_memory(memory: dict[str, Any]) -> None:
    normal = memory["normal_gpu_resident"]
    asym = memory["asym_cpu_resident"]
    print(
        "[M4 actual memory comparison] "
        f"normal_model_hbm={_fmt_mib(normal['model_hbm_bytes'])}, "
        f"asym_model_hbm={_fmt_mib(asym['model_hbm_bytes'])}, "
        f"hbm_model_saved={_fmt_mib(memory['hbm_model_saved_bytes'])}, "
        f"normal_peak_hbm={_fmt_mib(normal['peak_hbm_bytes'])}, "
        f"asym_peak_hbm={_fmt_mib(asym['peak_hbm_bytes'])}, "
        f"pinned_cpu={_fmt_mib(memory['pinned_cpu_bytes'])}, "
        f"expected_hbm_saved={_fmt_mib(memory['expected_hbm_saved_bytes'])}, "
        f"asym_stats={asym['execution_stats']}"
    )


def _route_op_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _manual_route_scatter(topk: torch.Tensor, weights: torch.Tensor, route_values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((topk.shape[0], route_values.shape[-1]), device=route_values.device, dtype=torch.float32)
    for token_idx in range(topk.shape[0]):
        for route_idx in range(topk.shape[1]):
            out[token_idx] += route_values[token_idx, route_idx].float() * weights[token_idx, route_idx]
    return out


@pytest.mark.parametrize("pattern", STATIC_PATTERNS)
@pytest.mark.parametrize("mode", GROUPED_MODES)
def test_tiny_moe_route_pack_scatter_ops_cover_metadata_modes(pattern: str, mode: str) -> None:
    config = MICRO_MOE_CONFIG
    device = _route_op_device()
    topk, weights = tiny_moe.make_static_routes(config, device, pattern=pattern)[0]
    num_tokens = _route_token_count(config)
    hidden = torch.arange(num_tokens * 5, device=device, dtype=torch.float32).reshape(num_tokens, 5)
    route_values = torch.arange(num_tokens * config.top_k * 5, device=device, dtype=torch.float32).reshape(
        num_tokens,
        config.top_k,
        5,
    )

    metadata = tiny_moe.build_route_metadata(topk, weights, num_experts=config.num_experts, mode=mode)
    summary = tiny_moe.route_metadata_summary(metadata)
    assert summary["active_routes"] == num_tokens * config.top_k

    if mode == "contiguous":
        assert isinstance(metadata, tiny_moe.ContiguousRouteMetadata)
        packed = tiny_moe.pack_tokens_contiguous(hidden, metadata)
        torch.testing.assert_close(packed, hidden.index_select(0, metadata.token_indices))
        sorted_values = route_values.reshape(-1, 5).index_select(0, metadata.route_indices)
        scattered = tiny_moe.scatter_contiguous(sorted_values, metadata)
    else:
        assert isinstance(metadata, tiny_moe.MaskedRouteMetadata)
        packed = tiny_moe.pack_tokens_masked(hidden, metadata)
        assert bool((packed[~metadata.valid_mask] == 0).all())
        masked_values = torch.zeros((*metadata.token_indices.shape, 5), device=device, dtype=torch.float32)
        masked_values[metadata.valid_mask] = route_values.reshape(-1, 5).index_select(
            0,
            metadata.route_indices[metadata.valid_mask],
        )
        scattered = tiny_moe.scatter_masked(masked_values, metadata)

    torch.testing.assert_close(scattered, _manual_route_scatter(topk, weights, route_values))


@pytest.mark.parametrize("pattern", STATIC_PATTERNS)
@pytest.mark.parametrize("mode", GROUPED_MODES)
def test_tiny_moe_scatter_backward_matches_autograd_and_repeated_backward(pattern: str, mode: str) -> None:
    config = MICRO_MOE_CONFIG
    device = _route_op_device()
    topk, weights = tiny_moe.make_static_routes(config, device, pattern=pattern)[0]
    weights = weights.detach().clone().requires_grad_(True)
    num_tokens = _route_token_count(config)
    feature = 7
    grad_seed = torch.linspace(-0.4, 0.6, num_tokens * feature, device=device, dtype=torch.float32).reshape(
        num_tokens,
        feature,
    )

    metadata = tiny_moe.build_route_metadata(topk, weights, num_experts=config.num_experts, mode=mode)
    if mode == "contiguous":
        assert isinstance(metadata, tiny_moe.ContiguousRouteMetadata)
        expert_output = torch.randn(metadata.num_routes, feature, device=device, dtype=torch.float32, requires_grad=True)
        out = tiny_moe.scatter_contiguous(expert_output, metadata)
        expected_expert_grad, expected_weight_grad = tiny_moe.scatter_backward_contiguous(
            grad_seed,
            expert_output.detach(),
            metadata,
        )
        expected_weight_grad = tiny_moe.restore_contiguous_route_order(expected_weight_grad, metadata)
    else:
        assert isinstance(metadata, tiny_moe.MaskedRouteMetadata)
        expert_output = torch.randn(
            config.num_experts,
            metadata.max_routes_per_expert,
            feature,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        out = tiny_moe.scatter_masked(expert_output, metadata)
        expected_expert_grad, expected_weight_grad = tiny_moe.scatter_backward_masked(
            grad_seed,
            expert_output.detach(),
            metadata,
        )
        expected_weight_grad = tiny_moe.restore_masked_route_order(expected_weight_grad, metadata)

    out.backward(grad_seed, retain_graph=True)
    out.backward(grad_seed)
    torch.testing.assert_close(expert_output.grad, expected_expert_grad * 2.0)
    torch.testing.assert_close(weights.grad, expected_weight_grad * 2.0)


@pytest.mark.parametrize("pattern", STATIC_PATTERNS)
@pytest.mark.parametrize("mode", GROUPED_MODES)
def test_tiny_moe_cpu_static_routing_patterns_and_grouped_modes(pattern: str, mode: str) -> None:
    case = _run_static_parity(pattern, mode)
    _print_static(case)

    assert case["output_max_abs"] < 1e-6
    assert case["loss_abs"] < 1e-6
    assert case["input_grad_max_abs"] < 1e-6
    assert case["lora_grad_worst_max_abs"] < 1e-6
    assert case["execution_stats"]["asym_calls"] == 0
    assert case["execution_stats"]["torch_calls"] > 0

    metadata = case["route_metadata"]
    if pattern == "balanced":
        assert metadata["empty_experts"] == 0
        if mode == "contiguous":
            assert metadata["padded_routes"] == 0
        else:
            assert metadata["padded_routes"] >= 0
    elif pattern == "empty":
        assert metadata["empty_experts"] > 0
        assert case["missing_lora_grad_names"]
    elif pattern == "skewed":
        counts = metadata["expert_counts"]
        assert counts[0] > counts[1] > counts[2] > counts[3]
    elif pattern == "repeated":
        assert metadata["active_routes"] == _route_token_count(MICRO_MOE_CONFIG) * MICRO_MOE_CONFIG.top_k
        assert case["repeated_token_expert_pairs"] == _route_token_count(MICRO_MOE_CONFIG)
    if pattern != "empty":
        assert case["missing_lora_grad_names"] == []


@pytest.mark.parametrize("mode", GROUPED_MODES)
def test_tiny_moe_cpu_learned_router_gradients_match_torch(mode: str) -> None:
    case = _run_learned_router_parity(mode)
    _print_learned(case)

    assert case["output_max_abs"] < 1e-6
    assert case["loss_abs"] < 1e-6
    assert case["input_grad_max_abs"] < 1e-6
    assert case["lora_grad_worst_max_abs"] < 1e-6
    assert case["router_grad_worst_max_abs"] < 1e-6
    assert case["execution_stats"]["asym_calls"] == 0
    assert case["execution_stats"]["torch_calls"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for trainable-state placement checks")
def test_tiny_moe_cuda_trainable_state_on_gpu_and_frozen_experts_on_cpu() -> None:
    config = MICRO_MOE_CONFIG
    device = torch.device("cuda")
    dtype = torch.bfloat16
    backend = "asym_only" if _direct_bf16_available() else "asym_or_staged"
    model, _, _, _ = tiny_moe.make_tiny_moe_pair(
        config=config,
        seed=330,
        device=device,
        base_dtype=dtype,
        backend=backend,
        pin_memory=True,
    )
    _assert_transformer_moe_modules(model, config)

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(_is_trainable_moe_name(name) for name, _ in trainable)
    assert any(_is_lora_name(name) for name, _ in trainable)
    assert any(_is_router_name(name) for name, _ in trainable)
    assert all(param.device.type == "cuda" for _, param in trainable)

    host_weights = _host_weight_items(model)
    host_names = [name for name, _ in host_weights]
    named_param_ids = {id(param) for _, param in model.named_parameters()}
    assert host_weights
    assert any("shared" in name.lower() for name in host_names), "shared expert base weights must use host storage"
    assert any("expert" in name.lower() and "shared" not in name.lower() for name in host_names)
    for _, weight in host_weights:
        assert weight.device.type == "cpu"
        assert not weight.requires_grad
        assert weight.grad is None
        assert id(weight) not in named_param_ids

    if _forward_accepts_transformer_inputs(model):
        inputs, labels = _make_transformer_inputs(config, device=device, dtype=dtype, seed=331)
        inputs = inputs.detach().clone().requires_grad_(True)
        output = _call_model(model, config=config, inputs=inputs, labels=labels, static_routing=None, mode="contiguous")
    else:
        inputs = _make_input(config, device=device, dtype=dtype, seed=331).requires_grad_(True)
        output = model(inputs, mode="contiguous")
    loss = _loss_from_output(output)
    loss.backward()
    _sync(device)

    active_trainable = [(name, param) for name, param in trainable if param.grad is not None]
    assert active_trainable
    assert any(_is_lora_name(name) for name, _ in active_trainable)
    assert any(_is_router_name(name) for name, _ in active_trainable)
    assert all(param.grad is not None and param.grad.device.type == "cuda" for _, param in active_trainable)


@pytest.mark.skipif(not _direct_bf16_available(), reason="direct-fetch tiny MoE requires SM90/SM100")
def test_tiny_moe_direct_fetch_correctness_hbm_and_flags(tmp_path) -> None:
    report = tiny_moe.run_tiny_moe_correctness_report(
        report_path=tmp_path / "m4_tiny_moe_direct_bf16.json",
        config=MICRO_MOE_CONFIG,
        device="cuda",
        backend="asym_only",
        seed=320,
    )
    _print_report(report)
    memory = report["memory_comparison"]
    _print_memory(memory)

    assert report["status"] == "pass"
    _assert_tiny_moe_report_contract(report)
    assert report["metadata_modes_tested"] == ["contiguous", "masked"]
    assert set(report["route_patterns_tested"]) == {"balanced", "empty", "skewed", "repeated"}
    assert report["toy_training"]["steps"] == 20
    assert report["toy_training"]["all_losses_finite"]
    assert report["toy_training"]["used_static_coverage_step"]
    assert report["toy_training"]["used_learned_router_step"]
    optimizer_state = report["toy_training"]["optimizer_state"]
    assert optimizer_state["optimizer_class"] == "AdamW"
    assert optimizer_state["expected_kind"] == "lora_plus_router"
    assert optimizer_state["expected_lora_param_count"] > 0
    assert optimizer_state["expected_router_param_count"] > 0
    assert optimizer_state["trainable_params_match_expected_kind"]
    assert optimizer_state["all_expected_params_in_optimizer"]
    assert optimizer_state["only_expected_params_in_optimizer"]
    assert optimizer_state["state_for_all_expected_params"]
    assert optimizer_state["adam_moments_for_all_expected_params"]
    assert optimizer_state["state_entry_count"] == optimizer_state["expected_state_entry_count"]
    assert optimizer_state["unexpected_state_param_count"] == 0
    assert optimizer_state["unexpected_optimizer_param_count"] == 0
    assert optimizer_state["state_tensor_bytes"] > 0
    assert optimizer_state["non_finite_state_names"] == []
    frozen_summary = report["toy_training"]["frozen_host_weight_summary"]
    assert frozen_summary["host_weight_count"] > 0
    assert frozen_summary["all_cpu"]
    assert frozen_summary["all_requires_grad_false"]
    assert frozen_summary["all_grads_absent"]
    assert frozen_summary["all_unchanged"]
    assert frozen_summary["absent_from_named_parameters"]
    assert frozen_summary["absent_from_optimizer_params"]
    assert frozen_summary["absent_from_optimizer_state"]
    assert report["direct_fetch_forward_used"]
    assert report["direct_fetch_dx_used"]
    assert report["fallback_counts"]["staged_calls"] == 0
    assert report["fallback_counts"]["torch_calls"] == 0
    static_patterns = set()
    for case in report["parity"]:
        assert case["output_max_abs"] <= 0.5
        assert case["loss_abs"] <= 0.05
        assert case["input_grad_max_abs"] <= 0.05
        assert case["lora_grad_worst_max_abs"] <= 0.05
        assert case["stats"]["staged_calls"] == 0
        assert case["stats"]["torch_calls"] == 0
        assert case["stats"]["asym_forward_calls"] > 0
        assert case["stats"]["asym_dx_calls"] > 0
        if case["learned_router"]:
            assert case["route_pattern"] == "learned"
            assert case["router_grad_worst_max_abs"] <= 0.05
        else:
            static_patterns.add(case["route_pattern"])
    assert report["static_logits_max_abs"] <= 0.5
    assert report["static_loss_abs"] <= 0.05
    assert report["static_input_grad_max_abs"] <= 0.05
    assert report["learned_router_logits_max_abs"] <= 0.5
    assert report["learned_router_loss_abs"] <= 0.05
    assert report["learned_router_input_grad_max_abs"] <= 0.05
    assert report["learned_router_grad_worst_max_abs"] <= 0.05
    assert report["expert_lora_grad_worst_max_abs"] <= 0.05
    assert static_patterns == set(report["route_patterns_tested"])

    normal = memory["normal_gpu_resident"]
    asym = memory["asym_cpu_resident"]
    assert normal["model_hbm_bytes"] > asym["model_hbm_bytes"]
    assert memory["hbm_model_saved_bytes"] >= int(memory["expected_hbm_saved_bytes"] * 0.8)
    assert memory["pinned_cpu_bytes"] >= memory["expected_hbm_saved_bytes"]
    assert asym["execution_stats"]["asym_forward_calls"] > 0
    assert asym["execution_stats"]["asym_dx_calls"] > 0
    assert asym["execution_stats"]["staged_calls"] == 0
    assert asym["execution_stats"]["torch_calls"] == 0
