from __future__ import annotations

import pytest
import torch


def _native_api():
    pytest.importorskip("asym_gemm")
    import asym_gemm

    api = getattr(asym_gemm, "qwen3_gate_up_recompute_bwd_sm100_bf16_windowed", None)
    if api is None:
        pytest.skip("qwen3 windowed backward native API is not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    return api


def _build_case(device: torch.device):
    torch.manual_seed(123)
    e, h, i, r, r_down = 3, 16, 10, 4, 3
    offsets = [0, 2, 5]
    experts = [0, 2]
    m = offsets[-1]
    x = torch.randn((m, h), device=device, dtype=torch.bfloat16)
    dact = torch.randn((m, i), device=device, dtype=torch.bfloat16)
    dS_down = torch.randn((m, r_down), device=device, dtype=torch.bfloat16)
    down_mask_bool = torch.rand((m, i), device=device) >= 0.25
    gate_low_rank = torch.randn((m, r), device=device, dtype=torch.bfloat16)
    up_low_rank = torch.randn((m, r), device=device, dtype=torch.bfloat16)
    gate_lora_b = torch.randn((e, i, r), device=device, dtype=torch.bfloat16)
    up_lora_b = torch.randn((e, i, r), device=device, dtype=torch.bfloat16)
    gate_up_weight_cpu = torch.randn((e, 2 * i, h), dtype=torch.bfloat16).pin_memory()
    offsets_t = torch.tensor(offsets, device=device, dtype=torch.int32)
    experts_t = torch.tensor([*experts, -1], device=device, dtype=torch.int32)
    return (
        x,
        dact,
        dS_down,
        down_mask_bool,
        gate_low_rank,
        up_low_rank,
        gate_lora_b,
        up_lora_b,
        gate_up_weight_cpu,
        offsets_t,
        experts_t,
        offsets,
        experts,
    )


def _reference(case, lora_scale: float, down_dropout_p: float):
    (
        x,
        dact,
        dS_down,
        down_mask_bool,
        gate_low_rank,
        up_low_rank,
        gate_lora_b,
        up_lora_b,
        gate_up_weight_cpu,
        _offsets_t,
        _experts_t,
        offsets,
        experts,
    ) = case
    m, h = x.shape
    i = dact.shape[1]
    w_gate = gate_up_weight_cpu[:, :i, :].to(device=x.device, dtype=torch.float32)
    w_up = gate_up_weight_cpu[:, i:, :].to(device=x.device, dtype=torch.float32)
    gate = torch.empty((m, i), device=x.device, dtype=torch.float32)
    up = torch.empty((m, i), device=x.device, dtype=torch.float32)
    for group, expert in enumerate(experts):
        start, end = offsets[group], offsets[group + 1]
        gate[start:end] = (
            x[start:end].float() @ w_gate[expert].T
            + lora_scale * (gate_low_rank[start:end].float() @ gate_lora_b[expert].float().T)
        )
        up[start:end] = (
            x[start:end].float() @ w_up[expert].T
            + lora_scale * (up_low_rank[start:end].float() @ up_lora_b[expert].float().T)
        )
    sigmoid = torch.sigmoid(gate)
    act = torch.nn.functional.silu(gate) * up
    grad_gate = (dact.float() * up * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))).to(torch.bfloat16)
    grad_up = (dact.float() * torch.nn.functional.silu(gate)).to(torch.bfloat16)
    if down_dropout_p > 0.0:
        act_for_down = torch.where(down_mask_bool, act * (1.0 / (1.0 - down_dropout_p)), torch.zeros_like(act))
    else:
        act_for_down = act
    grad_down_A = torch.zeros(
        (gate_up_weight_cpu.shape[0], dS_down.shape[1], i),
        device=x.device,
        dtype=torch.float32,
    )
    dx = torch.empty((m, h), device=x.device, dtype=torch.float32)
    for group, expert in enumerate(experts):
        start, end = offsets[group], offsets[group + 1]
        dx[start:end] = grad_gate[start:end].float() @ w_gate[expert] + grad_up[start:end].float() @ w_up[expert]
        grad_down_A[expert] += dS_down[start:end].float().T @ act_for_down[start:end]
    return dx.to(torch.bfloat16), grad_gate, grad_up, grad_down_A.to(torch.bfloat16)


def test_qwen3_gate_up_windowed_bwd_native_tiny():
    api = _native_api()
    device = torch.device("cuda")
    case = _build_case(device)
    lora_scale = 0.5
    down_dropout_p = 0.25
    dx_ref, gate_ref, up_ref, down_a_ref = _reference(case, lora_scale, down_dropout_p)
    x, dact, dS_down, down_mask_bool, gate_lr, up_lr, gate_b, up_b, weight_cpu, offsets_t, experts_t, *_ = case
    import asym_gemm

    down_mask_packed = asym_gemm.pack_bool_mask_2d(down_mask_bool)
    dx, grad_gate, grad_up, grad_down_A, stats = api(
        x,
        dact,
        gate_lr,
        up_lr,
        gate_b,
        up_b,
        weight_cpu,
        offsets_t,
        experts_t,
        p=4,
        q=2,
        bm=2,
        bk=8,
        g_work=2,
        lora_scale=lora_scale,
        mode="cache_first_window",
        return_stats=True,
        dS_down_sel=dS_down,
        down_mask_packed=down_mask_packed,
        down_dropout_p=down_dropout_p,
    )
    torch.testing.assert_close(grad_gate, gate_ref, rtol=0, atol=0)
    torch.testing.assert_close(grad_up, up_ref, rtol=0, atol=0)
    torch.testing.assert_close(dx, dx_ref, rtol=0, atol=0)
    torch.testing.assert_close(grad_down_A, down_a_ref, rtol=2.0e-2, atol=1.0)
    assert stats["cpu_weight_stream_multiplier"] == pytest.approx(1.0)
    assert stats["old_selected_base_dx_rows"] == 0
    assert stats["new_selected_base_dx_rows"] == int(x.shape[0])
    assert stats["down_lora_A_enabled"] is True
