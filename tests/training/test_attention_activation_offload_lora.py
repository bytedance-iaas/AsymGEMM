from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import asym_gemm
from asym_gemm.training.activation_offload import clear_activation_offload_cpu_pool
from asym_gemm.training.attention_activation_offload import (
    AsymActivationOffloadLoRALinear,
    AttentionActivationOffloadContext,
    install_attention_saved_tensor_offload,
)
from asym_gemm.training.cpu_adam import AsymCPUAdamW
from asym_gemm.training.frozen_linear import AsymExecutionStats
from asym_gemm.training.host_weight import HostWeight
from asym_gemm.training.lora import AsymLoRALinear


def _sm100_bf16_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 10
        and hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous")
        and hasattr(asym_gemm, "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous")
    )


def _make_pair(*, bias: bool = True) -> tuple[AsymLoRALinear, AsymActivationOffloadLoRALinear]:
    torch.manual_seed(101)
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    bias_tensor = torch.randn(32, dtype=torch.bfloat16) if bias else None
    host = HostWeight(weight, pin_memory=False, clone=True)
    current = AsymLoRALinear.from_host_weight(
        host,
        bias=bias_tensor,
        rank=4,
        alpha=8.0,
        backend="torch",
        stats=AsymExecutionStats(),
        device=torch.device("cpu"),
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
    )
    offload = AsymActivationOffloadLoRALinear.from_host_weight(
        host,
        bias=bias_tensor,
        rank=4,
        alpha=8.0,
        backend="torch",
        stats=AsymExecutionStats(),
        device=torch.device("cpu"),
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
        projection_role="q_proj",
    )
    with torch.no_grad():
        offload.lora_a.copy_(current.lora_a)
        offload.lora_b.copy_(current.lora_b)
    return current, offload


class _MiniSdpaAttention(nn.Module):
    def __init__(self, hidden: int = 256, heads: int = 8, kv_heads: int = 2, head_dim: int = 32) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _hidden = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.heads * self.head_dim)
        return self.o_proj(out)


def test_linear_forward_matches_current_without_dropout() -> None:
    current, offload = _make_pair()
    x = torch.randn(3, 5, 16, dtype=torch.bfloat16)

    expected = current(x)
    actual = offload(x)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SDPA")
def test_attention_saved_tensor_offload_preserves_sdpa_backward_strides() -> None:
    torch.manual_seed(307)
    expected = _MiniSdpaAttention().to("cuda")
    actual = copy.deepcopy(expected).to("cuda")
    install_attention_saved_tensor_offload(actual, min_bytes=1024)
    expected.train()
    actual.train()
    x_expected = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_actual = x_expected.detach().clone().requires_grad_(True)

    expected_loss = expected(x_expected).float().square().mean()
    actual_loss = actual(x_actual).float().square().mean()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x_actual.grad, x_expected.grad, atol=0.0, rtol=0.0)
    for expected_param, actual_param in zip(expected.parameters(), actual.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=0.0, rtol=0.0)
    stats = actual._last_activation_offload_stats
    assert stats["attention_saved_tensor_offload"] is True
    assert stats["num_offloads"] > 0
    assert stats["num_stages"] == stats["num_offloads"]
    assert stats["cpu_live_bytes"] == 0


def test_attention_saved_tensor_offload_dtype_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_DTYPES", "float32")
    module = _MiniSdpaAttention()

    wrapper = install_attention_saved_tensor_offload(module, min_bytes=1024)

    assert wrapper.allowed_dtypes == frozenset({torch.float32})
    assert module._last_activation_offload_stats["allowed_dtypes"] == ["float32"]


def test_linear_backward_matches_current_without_dropout() -> None:
    clear_activation_offload_cpu_pool()
    current, offload = _make_pair()
    x_expected = torch.randn(7, 16, dtype=torch.bfloat16, requires_grad=True)
    x_actual = x_expected.detach().clone().requires_grad_(True)

    expected_loss = current(x_expected).float().square().mean()
    actual_loss = offload(x_actual).float().square().mean()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x_actual.grad, x_expected.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(offload.lora_a.grad, current.lora_a.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(offload.lora_b.grad, current.lora_b.grad, atol=4e-2, rtol=4e-2)
    assert offload.base_layer.host_weight.weight.grad is None


def test_linear_preserves_frozen_bias() -> None:
    current, offload = _make_pair(bias=True)
    x = torch.randn(11, 16, dtype=torch.bfloat16)

    torch.testing.assert_close(offload(x), current(x), atol=0.0, rtol=0.0)


def test_linear_forward_and_backward_counts() -> None:
    _current, offload = _make_pair()
    stats = offload.base_layer.stats
    x = torch.randn(9, 16, dtype=torch.bfloat16, requires_grad=True)

    offload(x).float().square().mean().backward()

    assert stats.torch_forward_calls == 1
    assert stats.torch_dx_calls == 2
    assert stats.attn_act_base_dx_calls == 1
    assert stats.attn_act_lora_a_forward_calls == 1
    assert stats.attn_act_lora_a_grad_calls == 1
    assert stats.attn_act_stage_low_rank_calls == 1
    assert stats.attn_act_hbm_forward_calls == 2
    assert stats.attn_act_hbm_backward_calls == 3
    assert stats.forward_calls_total == 3
    assert stats.backward_calls_total == 6
    assert stats.calls_total == 9
    assert stats.attn_act_hbm_gemm_calls_by_tag == {
        "q_proj.base_forward": 1,
        "q_proj.lora_a_forward": 1,
        "q_proj.lora_b_forward": 1,
        "q_proj.dS": 1,
        "q_proj.base_dx": 1,
        "q_proj.lora_input_grad": 1,
        "q_proj.dA": 1,
        "q_proj.dB": 1,
    }
    assert offload._last_activation_offload_stats["cpu_live_bytes"] == 0
    assert offload._last_activation_offload_stats["staged_bytes"] == 0
    assert offload._last_activation_offload_stats["num_stages"] == 1


def test_linear_state_dict_keys_match_current_asym_lora() -> None:
    current, offload = _make_pair()

    assert set(offload.state_dict()) == set(current.state_dict())


def test_linear_rejects_dropout_until_supported() -> None:
    weight = torch.randn(16, 16, dtype=torch.bfloat16)
    module = AsymActivationOffloadLoRALinear.from_host_weight(
        HostWeight(weight, pin_memory=False, clone=True),
        rank=4,
        alpha=8.0,
        backend="torch",
        device=torch.device("cpu"),
        lora_dropout=0.1,
    )

    with pytest.raises(NotImplementedError, match="dropout"):
        module(torch.randn(5, 16, dtype=torch.bfloat16))


def test_cpu_adam_contract_updates_only_attention_lora_params() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(16, 16, dtype=torch.bfloat16)
    module = AsymActivationOffloadLoRALinear.from_host_weight(
        HostWeight(weight, pin_memory=False, clone=True),
        rank=4,
        alpha=8.0,
        backend="torch",
        device=device,
        lora_dropout=0.0,
        projection_role="q_proj",
    )
    x = torch.randn(6, 16, device=device, dtype=torch.bfloat16)
    optimizer = AsymCPUAdamW(
        list(module.named_parameters()),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        backend="torch",
        pin_memory=False,
    )
    before = {name: param.detach().clone() for name, param in module.named_parameters()}

    module(x).float().square().mean().backward()
    optimizer.step()

    assert optimizer.param_names == ["lora_A.default.weight", "lora_B.default.weight"]
    assert all("base_layer" not in name for name in optimizer.param_names)
    for name, param in module.named_parameters():
        assert not torch.equal(param.detach().cpu(), before[name].detach().cpu())


@pytest.mark.skipif(not _sm100_bf16_available(), reason="requires SM100 BF16 AsymGEMM kernels")
def test_linear_sm100_asym_backend_forward_backward_matches_current_asym() -> None:
    torch.manual_seed(173)
    weight = torch.randn(64, 64, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.bfloat16)
    host_current = HostWeight(weight, pin_memory=True, clone=True)
    host_offload = HostWeight(weight, pin_memory=True, clone=True)
    current = AsymLoRALinear.from_host_weight(
        host_current,
        bias=bias,
        rank=8,
        alpha=16.0,
        backend="asym",
        stats=AsymExecutionStats(),
        device=torch.device("cuda:0"),
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
    )
    offload = AsymActivationOffloadLoRALinear.from_host_weight(
        host_offload,
        bias=bias,
        rank=8,
        alpha=16.0,
        backend="asym",
        stats=AsymExecutionStats(),
        device=torch.device("cuda:0"),
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
        projection_role="q_proj",
    )
    with torch.no_grad():
        offload.lora_a.copy_(current.lora_a)
        offload.lora_b.copy_(current.lora_b)

    x_current = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    x_offload = x_current.detach().clone().requires_grad_(True)
    current_loss = current(x_current).float().square().mean()
    offload_loss = offload(x_offload).float().square().mean()
    current_loss.backward()
    offload_loss.backward()

    torch.testing.assert_close(offload_loss, current_loss, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(x_offload.grad, x_current.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(offload.lora_a.grad, current.lora_a.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(offload.lora_b.grad, current.lora_b.grad, atol=5e-2, rtol=5e-2)
    assert offload.base_layer.stats.asym_forward_calls >= 1
    assert offload.base_layer.stats.attn_act_lora_a_forward_calls == 1
    assert offload.base_layer.stats.attn_act_lora_a_grad_calls == 1
    assert offload._last_activation_offload_stats["cpu_live_bytes"] == 0


@pytest.mark.skipif(not _sm100_bf16_available(), reason="requires SM100 BF16 AsymGEMM kernels")
def test_qkv_wrappers_share_one_source_handle() -> None:
    torch.manual_seed(211)
    context = AttentionActivationOffloadContext()
    weight = torch.randn(64, 64, dtype=torch.bfloat16)
    modules = []
    for role in ("q_proj", "k_proj", "v_proj"):
        modules.append(
            AsymActivationOffloadLoRALinear.from_host_weight(
                HostWeight(weight, pin_memory=True, clone=True),
                rank=8,
                alpha=16.0,
                backend="asym",
                stats=AsymExecutionStats(),
                device=torch.device("cuda:0"),
                lora_dtype=torch.bfloat16,
                init_lora_weights="peft",
                lora_dropout=0.0,
                projection_role=role,
                attention_context=context,
            )
        )

    x = torch.randn(128, 64, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    loss = sum(module(x).float().square().mean() for module in modules)
    forward_snapshot = context.snapshot()
    loss.backward()
    backward_snapshot = context.snapshot()

    assert forward_snapshot["source_share_misses"] == 1
    assert forward_snapshot["source_share_hits"] == 2
    assert forward_snapshot["num_offloads"] == 1
    assert forward_snapshot["source_share_duplicate_bytes_avoided"] == 2 * x.numel() * x.element_size()
    assert forward_snapshot["source_share_cache_entries"] == 0
    assert backward_snapshot["cpu_live_bytes"] == 0
    assert backward_snapshot["source_share_released_bytes"] == x.numel() * x.element_size()
    assert all(module._last_activation_offload_stats["source_context"]["source_share_hits"] >= 2 for module in modules)
