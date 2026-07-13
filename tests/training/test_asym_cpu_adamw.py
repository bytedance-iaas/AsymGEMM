from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn

from asym_gemm.training.cpu_adam import AsymCPUAdamW
from asym_gemm.training.host_weight import HostWeight
from asym_gemm.training.lora import named_lora_parameters
from asym_gemm.training.offload import AsymFrozenEmbedding, AsymFrozenLayerNorm, AsymFrozenRMSNorm


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _device() -> torch.device:
    return torch.device("cuda:0" if _cuda_available() else "cpu")


class TinyLoRAModule(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False, device=device, dtype=dtype)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False, device=device, dtype=dtype)})
        self.base = nn.Linear(4, 3, bias=False, device=device, dtype=dtype)
        self.base.weight.requires_grad_(False)


def _fill_grads(named_params: list[tuple[str, nn.Parameter]], value: float = 0.125) -> None:
    for _, param in named_params:
        param.grad = torch.full_like(param, value)


def _lora_param_loss(named_params: list[tuple[str, nn.Parameter]], *, scale: float = 0.125) -> torch.Tensor:
    loss: torch.Tensor | None = None
    for index, (_, param) in enumerate(named_params, start=1):
        term = (param.float() * (scale * index)).sum()
        loss = term if loss is None else loss + term
    if loss is None:
        raise AssertionError("expected at least one LoRA parameter")
    return loss


def _assert_cpu_safe(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_cpu_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_cpu_safe(item)


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_torch_backend_updates_cuda_lora_params_and_keeps_state_on_cpu() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(
        named,
        lr=0.01,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.1,
        backend="torch",
        pin_memory=False,
    )
    ref_params = [nn.Parameter(param.detach().to(device="cpu", dtype=torch.float32).clone()) for _, param in named]
    ref_opt = torch.optim.AdamW(ref_params, lr=0.01, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.1)

    _fill_grads(named)
    for ref_param in ref_params:
        ref_param.grad = torch.full_like(ref_param, 0.125)

    opt.step()
    ref_opt.step()

    for mapping, ref_param in zip(opt._mappings, ref_params, strict=True):
        assert mapping.cpu_param.device.type == "cpu"
        assert mapping.cpu_param.dtype == torch.float32
        assert torch.allclose(mapping.cpu_param, ref_param, atol=1e-6, rtol=1e-5)
        expected_model = ref_param.detach().to(device=mapping.cuda_param.device, dtype=mapping.cuda_param.dtype)
        assert torch.allclose(mapping.cuda_param, expected_model, atol=0.0, rtol=0.0)
        assert mapping.cuda_param in opt.state
        assert opt.state[mapping.cuda_param]["cpu_master"].device.type == "cpu"
        assert opt.state[mapping.cuda_param]["exp_avg"].device.type == "cpu"
        assert opt.state[mapping.cuda_param]["exp_avg_sq"].device.type == "cpu"

    summary = opt.asym_cpu_adamw_summary()
    assert summary["param_count"] == len(named)
    assert summary["all_masters_on_cpu"] is True
    assert summary["all_cuda_params_on_cuda"] is True
    assert summary["optimizer_state_cpu_bytes"] > 0
    assert summary["last_step_grad_param_count"] == len(named)
    assert summary["last_step_copyback_param_count"] == len(named)


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_state_dict_load_restores_cpu_masters_and_cuda_params() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(named, lr=0.02, betas=(0.7, 0.95), eps=1e-5, weight_decay=0.01, backend="torch", pin_memory=False)
    _fill_grads(named, value=0.25)
    opt.step()
    saved = opt.state_dict()
    _assert_cpu_safe(saved)

    model2 = TinyLoRAModule(device=_device())
    with torch.no_grad():
        for _, param in named_lora_parameters(model2):
            param.zero_()
    opt2 = AsymCPUAdamW(
        named_lora_parameters(model2),
        lr=0.5,
        betas=(0.1, 0.2),
        eps=1e-3,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
    )

    moved_saved = copy.deepcopy(saved)
    moved_saved["cpu_master_params"] = [tensor.to(device=_device()) for tensor in moved_saved["cpu_master_params"]]
    opt2.load_state_dict(moved_saved)

    for mapping1, mapping2 in zip(opt._mappings, opt2._mappings, strict=True):
        assert mapping2.cpu_param.device.type == "cpu"
        assert torch.allclose(mapping2.cpu_param, mapping1.cpu_param)
        expected_model = mapping1.cpu_param.detach().to(device=mapping2.cuda_param.device, dtype=mapping2.cuda_param.dtype)
        assert torch.allclose(mapping2.cuda_param, expected_model, atol=0.0, rtol=0.0)
        assert opt2.state[mapping2.cuda_param]["exp_avg"].device.type == "cpu"
    assert opt2.param_groups[0]["lr"] == pytest.approx(saved["param_groups"][0]["lr"])


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_scheduler_lr_mutation_propagates_to_inner_optimizer() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(named, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, backend="torch", pin_memory=False)
    opt.param_groups[0]["lr"] = 0.003
    _fill_grads(named)

    opt.step()

    assert opt.inner_optimizer.param_groups[0]["lr"] == pytest.approx(0.003)


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_no_grad_param_skips_copyback_and_cpu_grad() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(named, lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, backend="torch", pin_memory=False)
    first_name, first_param = named[0]
    second_mapping = opt._mappings[1]
    before_cuda = second_mapping.cuda_param.detach().clone()
    before_cpu = second_mapping.cpu_param.detach().clone()
    first_param.grad = torch.full_like(first_param, 0.5)

    opt.step()

    assert opt.asym_cpu_adamw_summary()["last_step_grad_param_count"] == 1
    assert opt.asym_cpu_adamw_summary()["last_step_copyback_param_count"] == 1
    assert opt.asym_cpu_adamw_summary()["skipped_copyback_no_grad_param_count"] == len(named) - 1
    assert opt._mappings[0].name == first_name
    assert second_mapping.cpu_param.grad is None
    assert torch.equal(second_mapping.cuda_param, before_cuda)
    assert torch.equal(second_mapping.cpu_param, before_cpu)


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_duplicate_lora_params_create_one_cpu_master_with_alias_names() -> None:
    class SharedLoRA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(4, 2, bias=False, device=_device(), dtype=torch.bfloat16)
            self.lora_A = nn.ModuleDict({"default": shared})
            self.alias_lora_A = nn.ModuleDict({"default": shared})

    model = SharedLoRA()
    named = named_lora_parameters(model)
    assert len(named) == 2

    opt = AsymCPUAdamW(named, lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, backend="torch", pin_memory=False)

    assert len(opt.asym_cpu_master_params()) == 1
    assert opt.param_names == ["lora_A.default.weight"]
    assert opt.alias_param_names == [("alias_lora_A.default.weight",)]


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_different_views_on_same_storage_are_not_collapsed() -> None:
    class SlicedLoRA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.randn(8, device=_device(), dtype=torch.bfloat16)
            self.foo_lora_A = nn.Parameter(base[:4])
            self.bar_lora_A = nn.Parameter(base[4:])

    model = SlicedLoRA()
    opt = AsymCPUAdamW(
        list(model.named_parameters(remove_duplicate=False)),
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
    )

    assert len(opt.asym_cpu_master_params()) == 2


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_rejects_non_lora_and_cpu_resident_lora_params() -> None:
    dense = nn.Parameter(torch.ones(2, 2, device=_device(), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="only LoRA"):
        AsymCPUAdamW(
            [("dense.weight", dense)],
            lr=0.01,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            backend="torch",
            pin_memory=False,
        )

    cpu_lora = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="Stage 7"):
        AsymCPUAdamW(
            [("layer.lora_A.default.weight", cpu_lora)],
            lr=0.01,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            backend="torch",
            pin_memory=False,
        )


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_named_lora_params_ignore_frozen_base_offload_owners() -> None:
    class FrozenBasePlusLoRA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = AsymFrozenEmbedding(nn.Embedding(8, 4, dtype=torch.bfloat16))
            self.layer_norm = AsymFrozenLayerNorm(nn.LayerNorm(4, dtype=torch.bfloat16))
            rms = nn.Module()
            rms.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
            rms.variance_epsilon = 1e-6
            self.rms_norm = AsymFrozenRMSNorm(rms)
            self.host = HostWeight(torch.randn(4, 4, dtype=torch.bfloat16), pin_memory=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False, device=_device(), dtype=torch.bfloat16)})

    model = FrozenBasePlusLoRA()
    named = named_lora_parameters(model)
    assert [name for name, _ in named] == ["lora_A.default.weight"]

    opt = AsymCPUAdamW(named, lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, backend="torch", pin_memory=False)

    assert len(opt.asym_cpu_master_params()) == 1
    assert opt.param_names == ["lora_A.default.weight"]


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW runtime tests require CUDA")
def test_deepspeed_backend_one_step_if_extension_is_available() -> None:
    try:
        import deepspeed.ops.adam  # noqa: F401
    except Exception as exc:
        pytest.skip(f"DeepSpeedCPUAdam import unavailable: {exc}")

    model = TinyLoRAModule(device=_device())
    try:
        opt = AsymCPUAdamW(
            named_lora_parameters(model),
            lr=0.01,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            backend="deepspeed",
            pin_memory=False,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))

    _fill_grads(named_lora_parameters(model))
    opt.step()

    assert opt.asym_cpu_adamw_summary()["last_step_grad_param_count"] == len(named_lora_parameters(model))


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW grad offload tests require CUDA")
def test_grad_offload_hook_clears_cuda_grads_and_matches_step_copy_path() -> None:
    torch.manual_seed(0)
    model = TinyLoRAModule(device=_device())
    reference = TinyLoRAModule(device=_device())
    reference.load_state_dict(model.state_dict())
    named = named_lora_parameters(model)
    reference_named = named_lora_parameters(reference)

    offload_opt = AsymCPUAdamW(
        named,
        lr=0.01,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.1,
        backend="torch",
        pin_memory=False,
        grad_offload=True,
    )
    reference_opt = AsymCPUAdamW(
        reference_named,
        lr=0.01,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.1,
        backend="torch",
        pin_memory=False,
        grad_offload=False,
    )

    _lora_param_loss(named).backward()
    _lora_param_loss(reference_named).backward()

    assert all(param.grad is None for _, param in named)
    assert all(param.grad is not None for _, param in reference_named)
    assert len(offload_opt.asym_cpu_adamw_grad_buffers()) == len(named)
    assert all(mapping.grad_buffer_has_data for mapping in offload_opt._mappings)

    offload_opt.step()
    reference_opt.step()

    for offload_mapping, reference_mapping in zip(offload_opt._mappings, reference_opt._mappings, strict=True):
        assert torch.allclose(offload_mapping.cpu_param, reference_mapping.cpu_param, atol=1e-6, rtol=1e-5)
        assert torch.equal(offload_mapping.cuda_param, reference_mapping.cuda_param)
        assert offload_mapping.cuda_param.grad is None

    summary = offload_opt.asym_cpu_adamw_summary()
    assert summary["grad_offload_enabled"] is True
    assert summary["last_step_used_offloaded_grads"] is True
    assert summary["last_step_grad_param_count"] == len(named)
    assert summary["last_hook_call_count"] == len(named)
    assert summary["last_hook_offloaded_param_count"] == len(named)
    assert summary["last_hook_offloaded_numel"] == sum(param.numel() for _, param in named)
    assert summary["step_grad_copy_ms"] == pytest.approx(0.0)
    assert summary["grad_offload_buffer_bytes"] == sum(param.numel() for _, param in named) * 4


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW grad offload tests require CUDA")
def test_grad_offload_accumulates_multiple_backwards_before_step() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(
        named,
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
        grad_offload=True,
    )

    _lora_param_loss(named, scale=0.25).backward()
    _lora_param_loss(named, scale=0.25).backward()

    assert all(param.grad is None for _, param in named)
    for index, mapping in enumerate(opt._mappings, start=1):
        assert mapping.grad_buffer is not None
        assert mapping.grad_buffer_has_data is True
        assert mapping.hook_calls == 2
        assert mapping.offloaded_grad_numel == mapping.cuda_param.numel() * 2
        expected = torch.full_like(mapping.grad_buffer, 0.5 * index)
        assert torch.allclose(mapping.grad_buffer, expected)


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW grad offload tests require CUDA")
def test_grad_offload_zero_grad_clears_cpu_grad_ownership() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(
        named,
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
        grad_offload=True,
    )

    _lora_param_loss(named).backward()
    assert len(opt.asym_cpu_adamw_grad_buffers()) == len(named)

    opt.zero_grad()

    assert all(param.grad is None for _, param in named)
    assert opt.asym_cpu_adamw_grad_buffers() == []
    for mapping in opt._mappings:
        assert mapping.cpu_param.grad is None
        assert mapping.grad_buffer_has_data is False
        assert mapping.last_had_grad is False
        assert mapping.hook_calls == 0
        assert mapping.offloaded_grad_numel == 0


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW grad offload tests require CUDA")
def test_grad_offload_clip_grad_norm_scales_cpu_buffers_without_cuda_grads() -> None:
    model = TinyLoRAModule(device=_device())
    named = named_lora_parameters(model)
    opt = AsymCPUAdamW(
        named,
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
        grad_offload=True,
    )

    _lora_param_loss(named, scale=1.0).backward()
    before_norm = opt.asym_cpu_adamw_grad_norm()
    total_norm, summary = opt.asym_cpu_adamw_clip_grad_norm_(0.5)
    after_norm = opt.asym_cpu_adamw_grad_norm()

    assert float(total_norm.item()) == pytest.approx(float(before_norm.item()))
    assert float(before_norm.item()) > 0.5
    assert float(after_norm.item()) <= 0.50001
    assert all(param.grad is None for _, param in named)
    assert summary["path"] == "asym_cpu_adamw_grad_offload"
    assert summary["operation"] == "clip"
    assert summary["clipped"] is True
    assert summary["nonfinite"] is False
    assert summary["clip_coef_clamped"] == pytest.approx(summary["clip_coef"])
    assert summary["result_norm"] <= 0.50001
    assert summary["cpu_grad_tensors"] == len(named)
    assert summary["cpu_grad_numel"] == sum(param.numel() for _, param in named)
    assert summary["cuda_grad_tensors"] == 0

    norm_only, norm_summary = opt.asym_cpu_adamw_clip_grad_norm_(float("inf"))
    assert float(norm_only.item()) == pytest.approx(float(after_norm.item()))
    assert norm_summary["operation"] == "norm_only"
    assert norm_summary["max_norm"] == "inf"
    assert norm_summary["clip_coef_clamped"] == pytest.approx(1.0)
    assert norm_summary["clipped"] is False

    with pytest.raises(ValueError, match="chunk_elements"):
        opt.asym_cpu_adamw_grad_norm(chunk_elements=0)


@pytest.mark.skipif(not _cuda_available(), reason="AsymCPUAdamW grad offload tests require CUDA")
def test_grad_offload_empty_step_reports_no_consumed_grads() -> None:
    model = TinyLoRAModule(device=_device())
    opt = AsymCPUAdamW(
        named_lora_parameters(model),
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
        grad_offload=True,
    )

    opt.step()

    summary = opt.asym_cpu_adamw_summary()
    assert summary["grad_offload_enabled"] is True
    assert summary["last_step_used_offloaded_grads"] is False
    assert summary["last_step_grad_param_count"] == 0
    assert summary["skipped_copyback_no_grad_param_count"] == len(opt._mappings)
