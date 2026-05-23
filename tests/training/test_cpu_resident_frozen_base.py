import importlib
from types import SimpleNamespace

import pytest
import torch

import asym_gemm
from asym_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_nvfp4_e4m3
from asym_gemm.training import (
    AsymExecutionStats,
    AsymFrozenLinear,
    AsymGroupedFrozenLinear,
    HostWeight,
    VALID_BACKENDS,
    asym_frozen_linear,
    asym_grouped_frozen_linear,
    can_use_direct_bf16,
    measure_gpu_weight_allocation,
)

frozen_linear_impl = importlib.import_module("asym_gemm.training.frozen_linear")
lora_impl = importlib.import_module("asym_gemm.training.lora")


def _direct_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] in {9, 10}


def _direct_precision_available(precision: str) -> bool:
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_capability(0)[0]
    if precision == "fp8":
        return arch in {9, 10} and hasattr(asym_gemm, "m_grouped_fp8_asym_gemm_nt_contiguous")
    if precision == "fp4":
        return arch == 10 and hasattr(asym_gemm, "m_grouped_fp4_asym_gemm_nt_contiguous")
    return False


def _torch_grouped_mm_available() -> bool:
    return torch.cuda.is_available()


def _relative_max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float()
    b_f = b.detach().float()
    denom = float(b_f.abs().max().clamp_min(1e-6).item())
    return float((a_f - b_f).abs().max().item() / denom)


def _expand_last_dim_scales(scales: torch.Tensor, cols: int, gran_k: int) -> torch.Tensor:
    return scales.float().repeat_interleave(gran_k, dim=-1)[..., :cols]


def _expand_fp8_weight_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return scales.float().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[..., :rows, :cols]


def _unpack_fp4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.uint8)
    codes[..., 0::2] = lo
    codes[..., 1::2] = hi
    return codes[..., :cols]


def _decode_fp4_e2m1(codes: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
    magnitude = (codes & 0x07).long()
    decoded = levels[magnitude]
    sign = ((codes & 0x08) != 0) & (magnitude != 0)
    return torch.where(sign, -decoded, decoded)


def _quantized_linear_reference(a: torch.Tensor, host_weight: HostWeight, precision: str, *, transpose: bool = False) -> torch.Tensor:
    qweight = frozen_linear_impl._get_quantized_host_weight(host_weight, precision, transpose=transpose)
    assert qweight is not None
    k = int(qweight.logical_shape[-1])
    n = int(qweight.logical_shape[-2])
    if precision == "fp8":
        a_values, a_scales = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
        a_deq = a_values.float() * _expand_last_dim_scales(a_scales, k, 128)
        b_values = qweight.values.to(device=a.device)
        b_scales = qweight.scales.to(device=a.device)
        b_deq = b_values.float() * _expand_fp8_weight_scales(b_scales, n, k)
    elif precision == "fp4":
        a_values, a_scales = per_token_cast_to_nvfp4_e4m3(a, gran_k=16)
        a_deq = _decode_fp4_e2m1(_unpack_fp4(a_values, k)).float() * _expand_last_dim_scales(a_scales, k, 16)
        b_values = qweight.values.to(device=a.device)
        b_scales = qweight.scales.to(device=a.device)
        b_deq = _decode_fp4_e2m1(_unpack_fp4(b_values, k)).float() * _expand_last_dim_scales(b_scales, k, 16)
    else:
        raise ValueError(f"unexpected precision={precision!r}")
    return a_deq.float() @ b_deq.float().t()


def test_fp64_reference_gradcheck_cpu() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.float64)
    host_weight = HostWeight.from_tensor(weight, pin_memory=False)

    def func(inp: torch.Tensor) -> torch.Tensor:
        return asym_frozen_linear(inp, host_weight, backend="torch")

    assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)


def test_bias_grad_matches_torch_for_batched_inputs_cpu() -> None:
    torch.manual_seed(10)
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(7, 5, dtype=torch.float64)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    host_weight = HostWeight.from_tensor(weight, pin_memory=False)

    y = asym_frozen_linear(x, host_weight, bias=bias, backend="torch")
    y_ref = torch.nn.functional.linear(x_ref, weight, bias_ref)
    loss = (y.square().mean() + y[..., :2].sum() * 0.03)
    loss_ref = (y_ref.square().mean() + y_ref[..., :2].sum() * 0.03)
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(bias.grad, bias_ref.grad)
    assert host_weight.weight.grad is None


def test_backward_does_not_materialize_host_weight_transpose_cpu() -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.float32)
    host_weight = HostWeight.from_tensor(weight, pin_memory=False)

    y = asym_frozen_linear(x, host_weight, backend="torch")
    y.float().square().mean().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = x_ref @ weight.t()
    y_ref.float().square().mean().backward()

    assert host_weight.pinned_cpu_bytes == 0
    torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinning required")
def test_host_weight_exposes_only_single_pinned_weight_copy() -> None:
    torch.manual_seed(12)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    host_weight = HostWeight.from_tensor(weight, pin_memory=True)
    if not host_weight.weight.is_pinned():
        pytest.skip("pin_memory unavailable")

    assert host_weight.pinned_cpu_bytes == host_weight.weight_nbytes


def test_grouped_frozen_linear_cpu_forward_dx_and_no_dw() -> None:
    torch.manual_seed(13)
    x = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(3, 7, 5, dtype=torch.float64)
    grouped = AsymGroupedFrozenLinear(weight, backend="torch", pin_memory=False)
    offsets = torch.tensor([0, 2, 5, 9], dtype=torch.long)
    experts = torch.tensor([0, 1, 2, -1], dtype=torch.long)

    y = grouped(x, offsets, experts)
    y_ref = torch.cat(
        [
            x_ref[0:2] @ weight[0].t(),
            x_ref[2:5] @ weight[1].t(),
            x_ref[5:9] @ weight[2].t(),
        ],
        dim=0,
    )
    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert grouped.host_weight.weight.device.type == "cpu"
    assert grouped.host_weight.weight.grad is None


def test_grouped_frozen_linear_empty_group_cpu() -> None:
    torch.manual_seed(14)
    x = torch.randn(6, 5, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(4, 7, 5, dtype=torch.float32)
    host_weight = HostWeight(weight, pin_memory=False, clone=True, require_2d=False)
    offsets = torch.tensor([0, 2, 2, 4, 6], dtype=torch.long)
    experts = torch.tensor([0, 1, 2, 3, -1], dtype=torch.long)

    y = asym_grouped_frozen_linear(x, host_weight, offsets, experts, backend="torch")
    y_ref = torch.cat(
        [
            x_ref[0:2] @ weight[0].t(),
            x_ref[2:4] @ weight[2].t(),
            x_ref[4:6] @ weight[3].t(),
        ],
        dim=0,
    )
    (y.float().sum() * 0.1).backward()
    (y_ref.float().sum() * 0.1).backward()

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert host_weight.weight.grad is None


@pytest.mark.skipif(not _torch_grouped_mm_available(), reason="torch grouped_mm requires CUDA and PyTorch grouped MM")
def test_torch_grouped_frozen_linear_cuda_uses_grouped_mm_forward_and_dx(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(19)
    k = 128
    n = 64
    counts = [128, 0, 256, 128]
    offsets = torch.tensor([0, 128, 128, 384, 512], device="cuda", dtype=torch.long)
    experts = torch.tensor([2, 0, 3, 1, -1], device="cuda", dtype=torch.long)
    x = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(4, n, k, device="cuda", dtype=torch.bfloat16)
    grouped = frozen_linear_impl.TorchGroupedFrozenLinear(weight, device=torch.device("cuda"), dtype=torch.bfloat16)

    real_grouped_mm = frozen_linear_impl._TORCH_GROUPED_MM
    calls: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype, bool]] = []

    def counted_grouped_mm(mat1: torch.Tensor, mat2: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(mat1.shape), tuple(mat2.shape), offs.dtype, mat2.is_contiguous()))
        return real_grouped_mm(mat1, mat2, offs=offs)

    monkeypatch.setattr(frozen_linear_impl, "_TORCH_GROUPED_MM", counted_grouped_mm)

    y = grouped(x, offsets, experts)
    y_ref = torch.cat(
        [
            x_ref[0:128] @ weight[2].t(),
            x_ref[128:384] @ weight[3].t(),
            x_ref[384:512] @ weight[1].t(),
        ],
        dim=0,
    )
    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.02, rtol=0.02)
    assert calls == [
        ((512, k), (3, k, n), torch.int32, False),
        ((512, n), (3, n, k), torch.int32, True),
    ]


@pytest.mark.skipif(not _torch_grouped_mm_available(), reason="torch grouped_mm requires CUDA and PyTorch grouped MM")
def test_torch_grouped_frozen_linear_dense_experts_avoids_weight_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(22)
    k = 128
    n = 64
    offsets = torch.tensor([0, 128, 128, 384, 512], device="cuda", dtype=torch.long)
    experts = torch.tensor([0, 1, 2, 3, -1], device="cuda", dtype=torch.long)
    x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(4, n, k, device="cuda", dtype=torch.bfloat16)
    grouped = frozen_linear_impl.TorchGroupedFrozenLinear(weight, device=torch.device("cuda"), dtype=torch.bfloat16)

    real_grouped_mm = frozen_linear_impl._TORCH_GROUPED_MM
    calls: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype, bool]] = []

    def counted_grouped_mm(mat1: torch.Tensor, mat2: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(mat1.shape), tuple(mat2.shape), offs.dtype, mat2.is_contiguous()))
        return real_grouped_mm(mat1, mat2, offs=offs)

    monkeypatch.setattr(frozen_linear_impl, "_TORCH_GROUPED_MM", counted_grouped_mm)

    y = grouped(x, offsets, experts, dense_experts=True)
    y_ref = torch.cat(
        [
            x_ref[0:128] @ weight[0].t(),
            x_ref[128:384] @ weight[2].t(),
            x_ref[384:512] @ weight[3].t(),
        ],
        dim=0,
    )
    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.02, rtol=0.02)
    assert calls == [
        ((512, k), (4, k, n), torch.int32, False),
        ((512, n), (4, n, k), torch.int32, True),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch grouped_mm error path")
def test_torch_grouped_frozen_linear_cuda_propagates_grouped_mm_shape_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20)
    x = torch.randn(5, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 6, 8, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.long)
    experts = torch.tensor([1, 0, -1], device="cuda", dtype=torch.long)

    def rejecting_grouped_mm(mat1: torch.Tensor, mat2: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("strides should be multiple of 16 bytes")

    monkeypatch.setattr(frozen_linear_impl, "_TORCH_GROUPED_MM", rejecting_grouped_mm)

    with pytest.raises(RuntimeError, match="strides should be multiple of 16 bytes"):
        frozen_linear_impl._grouped_torch_chunks(x, weight, offsets, experts)


@pytest.mark.skipif(not _torch_grouped_mm_available(), reason="torch grouped_mm requires CUDA and PyTorch grouped MM")
def test_grouped_expert_lora_cuda_uses_grouped_mm_without_transpose_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(21)
    k = 128
    n = 16
    offsets = torch.tensor([0, 128, 128, 384, 512], device="cuda", dtype=torch.long)
    experts = torch.tensor([2, 0, 3, 1, -1], device="cuda", dtype=torch.long)
    x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(4, n, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    real_grouped_mm = lora_impl._TORCH_GROUPED_MM
    calls: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype, bool]] = []

    def counted_grouped_mm(mat1: torch.Tensor, mat2: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(mat1.shape), tuple(mat2.shape), offs.dtype, mat2.is_contiguous()))
        return real_grouped_mm(mat1, mat2, offs=offs)

    monkeypatch.setattr(lora_impl, "_TORCH_GROUPED_MM", counted_grouped_mm)

    y = lora_impl.grouped_expert_lora(x, weight, offsets, experts)
    y_ref = torch.cat(
        [
            x_ref[0:128] @ weight_ref[2].t(),
            x_ref[128:384] @ weight_ref[3].t(),
            x_ref[384:512] @ weight_ref[1].t(),
        ],
        dim=0,
    )
    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.02, rtol=0.02)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=0.02, rtol=0.02)
    assert calls == [((512, k), (3, k, n), torch.int32, False)]


@pytest.mark.skipif(not _torch_grouped_mm_available(), reason="torch grouped_mm requires CUDA and PyTorch grouped MM")
def test_grouped_expert_lora_dense_no_empty_uses_training_safe_full_stack() -> None:
    torch.manual_seed(25)
    k = 128
    n = 16
    offsets = torch.tensor([0, 128, 256, 384, 512], device="cuda", dtype=torch.long)
    experts = torch.tensor([0, 1, 2, 3, -1], device="cuda", dtype=torch.long)
    metadata = lora_impl.prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
    assert metadata.dense_expert_weights

    x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(4, n, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    y = lora_impl.grouped_expert_lora(x, weight, offsets, experts, metadata=metadata)
    y_ref = torch.cat(
        [
            x_ref[0:128] @ weight_ref[0].t(),
            x_ref[128:256] @ weight_ref[1].t(),
            x_ref[256:384] @ weight_ref[2].t(),
            x_ref[384:512] @ weight_ref[3].t(),
        ],
        dim=0,
    )
    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.02, rtol=0.02)
    torch.testing.assert_close(weight.grad, weight_ref.grad, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not _torch_grouped_mm_available(), reason="torch grouped_mm requires CUDA and PyTorch grouped MM")
def test_packed_expert_lora_gate_up_cuda_uses_two_grouped_mm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(24)
    hidden = 128
    intermediate = 64
    rank = 16
    num_experts = 4
    offsets = torch.tensor([0, 128, 128, 384, 512], device="cuda", dtype=torch.long)
    experts = torch.tensor([2, 0, 3, 1, -1], device="cuda", dtype=torch.long)

    def expert_state() -> dict[str, torch.Tensor]:
        return {
            "gate_lora_a": torch.randn(rank, hidden, dtype=torch.bfloat16) * 0.01,
            "gate_lora_b": torch.randn(intermediate, rank, dtype=torch.bfloat16) * 0.01,
            "up_lora_a": torch.randn(rank, hidden, dtype=torch.bfloat16) * 0.01,
            "up_lora_b": torch.randn(intermediate, rank, dtype=torch.bfloat16) * 0.01,
            "down_lora_a": torch.randn(rank, intermediate, dtype=torch.bfloat16) * 0.01,
            "down_lora_b": torch.randn(hidden, rank, dtype=torch.bfloat16) * 0.01,
        }

    states = [expert_state() for _ in range(num_experts)]
    config = SimpleNamespace(hidden_size=hidden, intermediate_size=intermediate, lora_scale=0.5)
    fused = lora_impl.PackedExpertLoRA(states, config=config, device=torch.device("cuda"), lora_dtype=torch.bfloat16)
    ref = lora_impl.PackedExpertLoRA(states, config=config, device=torch.device("cuda"), lora_dtype=torch.bfloat16)
    x = torch.randn(512, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    ref_metadata = ref.prepare_metadata(offsets, experts)
    gate_ref = ref(x_ref, offsets, experts, "gate", torch.bfloat16, metadata=ref_metadata)
    up_ref = ref(x_ref, offsets, experts, "up", torch.bfloat16, metadata=ref_metadata)
    loss_ref = gate_ref.float().square().mean() + up_ref.float().square().mean()
    loss_ref.backward()

    real_grouped_mm = lora_impl._TORCH_GROUPED_MM
    calls: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype, bool]] = []

    def counted_grouped_mm(mat1: torch.Tensor, mat2: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(mat1.shape), tuple(mat2.shape), offs.dtype, mat2.is_contiguous()))
        return real_grouped_mm(mat1, mat2, offs=offs)

    monkeypatch.setattr(lora_impl, "_TORCH_GROUPED_MM", counted_grouped_mm)

    metadata = fused.prepare_metadata(offsets, experts)
    gate, up = fused.forward_gate_up(x, offsets, experts, torch.bfloat16, metadata=metadata)
    loss = gate.float().square().mean() + up.float().square().mean()
    loss.backward()

    torch.testing.assert_close(gate, gate_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(up, up_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.02, rtol=0.02)
    for name in ("gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b"):
        torch.testing.assert_close(getattr(fused, name).grad, getattr(ref, name).grad, atol=0.02, rtol=0.02)
    assert calls == [
        ((512, hidden), (3, hidden, 2 * rank), torch.int32, False),
        ((1024, rank), (6, rank, intermediate), torch.int32, False),
    ]


def test_transpose_capability_check_uses_original_weight_without_copy() -> None:
    torch.manual_seed(12)
    x = torch.randn(3, 7, dtype=torch.float32)
    weight = torch.randn(7, 5, dtype=torch.float32)
    host_weight = HostWeight.from_tensor(weight, pin_memory=False)

    ok, reason = can_use_direct_bf16(x, host_weight, transpose=True)

    assert not ok
    assert reason in {"cuda_unavailable", "input_not_cuda", "requires_bf16"}
    assert host_weight.pinned_cpu_bytes == 0


@pytest.mark.skipif(not _direct_bf16_available(), reason="direct BF16 AsymGEMM requires SM90/SM100")
def test_direct_bf16_forward_and_dx_match_torch() -> None:
    torch.manual_seed(1)
    m = 17
    n = 24
    k = 16
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    weight_cuda = weight.to("cuda")
    host_weight = HostWeight.from_tensor(weight, pin_memory=True)
    stats = AsymExecutionStats()

    ok, reason = can_use_direct_bf16(x, host_weight)
    assert ok, reason

    y = asym_frozen_linear(x, host_weight, backend="asym", stats=stats)
    y_ref = x_ref @ weight_cuda.t()
    torch.testing.assert_close(y, y_ref, atol=0.0, rtol=0.0)

    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    forward_max_abs = float((y.float() - y_ref.float()).abs().max().item())
    dx_max_abs = float((x.grad.float() - x_ref.grad.float()).abs().max().item())
    print(
        "\n[M1 direct BF16 numerical error] "
        f"forward_max_abs={forward_max_abs:.6g}, dx_max_abs={dx_max_abs:.6g}, "
        f"asym_forward_calls={stats.asym_forward_calls}, asym_dx_calls={stats.asym_dx_calls}"
    )

    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.04, rtol=0.04)
    assert host_weight.pinned_cpu_bytes == host_weight.weight_nbytes
    assert stats.asym_forward_calls == 1
    assert stats.asym_dx_calls == 1
    assert stats.staged_calls == 0
    assert stats.torch_calls == 0


@pytest.mark.skipif(not _direct_bf16_available(), reason="direct grouped BF16 AsymGEMM requires SM90/SM100")
def test_direct_grouped_bf16_forward_and_dx_match_torch() -> None:
    torch.manual_seed(15)
    counts = [5, 7, 6]
    m = sum(counts)
    n = 64
    k = 64
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(len(counts), n, k, dtype=torch.bfloat16)
    weight_cuda = weight.to("cuda")
    grouped = AsymGroupedFrozenLinear(weight, backend="asym", pin_memory=True, stats=AsymExecutionStats())
    offsets = torch.tensor([0, counts[0], counts[0] + counts[1], m], device="cuda", dtype=torch.long)
    experts = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.long)

    y = grouped(x, offsets, experts)
    y_ref = torch.cat(
        [
            x_ref[0 : counts[0]] @ weight_cuda[0].t(),
            x_ref[counts[0] : counts[0] + counts[1]] @ weight_cuda[1].t(),
            x_ref[counts[0] + counts[1] :] @ weight_cuda[2].t(),
        ],
        dim=0,
    )
    loss = y.float().square().mean()
    loss_ref = y_ref.float().square().mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(y, y_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.04, rtol=0.04)
    assert grouped.host_weight.weight.device.type == "cpu"
    assert grouped.host_weight.weight.grad is None
    assert grouped.host_weight.pinned_cpu_bytes == grouped.host_weight.weight_nbytes
    assert grouped.stats.asym_forward_calls == 1
    assert grouped.stats.asym_dx_calls == 1
    assert grouped.stats.staged_calls == 0
    assert grouped.stats.torch_calls == 0


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_direct_quantized_precision_forward_and_dx_match_quantized_reference(precision: str) -> None:
    if not _direct_precision_available(precision):
        pytest.skip(f"direct {precision.upper()} AsymGEMM is not available on this device/build")

    torch.manual_seed(17)
    m = 128
    n = 128
    k = 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    stats = AsymExecutionStats()
    base = AsymFrozenLinear(weight, backend="asym", pin_memory=True, stats=stats, precision=precision)

    y = base(x)
    y_ref = _quantized_linear_reference(x, base.host_weight, precision)
    forward_diff = _relative_max_diff(y, y_ref)

    grad_out = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, grad_out)
    dx_ref = _quantized_linear_reference(grad_out.contiguous(), base.host_weight, precision, transpose=True)
    dx_diff = _relative_max_diff(dx, dx_ref)

    print(
        f"\n[M1 direct {precision.upper()} quantized error] "
        f"forward_rel_max={forward_diff:.6g}, dx_rel_max={dx_diff:.6g}, "
        f"asym_forward_calls={stats.asym_forward_calls}, asym_dx_calls={stats.asym_dx_calls}"
    )

    assert forward_diff < 3e-2
    assert dx_diff < 3e-2
    assert stats.asym_forward_calls == 1
    assert stats.asym_dx_calls == 1
    assert stats.staged_calls == 0
    assert stats.torch_calls == 0
    assert base.host_weight.weight.device.type == "cpu"
    assert base.pinned_cpu_bytes >= base.weight_hbm_saved_bytes


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_direct_grouped_quantized_precision_forward_and_dx_match_quantized_reference(precision: str) -> None:
    if not _direct_precision_available(precision):
        pytest.skip(f"direct grouped {precision.upper()} AsymGEMM is not available on this device/build")

    torch.manual_seed(19)
    counts = [128, 128]
    m = sum(counts)
    n = 128
    k = 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(len(counts), n, k, dtype=torch.bfloat16)
    stats = AsymExecutionStats()
    grouped = AsymGroupedFrozenLinear(weight, backend="asym", pin_memory=True, stats=stats, precision=precision)
    offsets = torch.tensor([0, counts[0], m], device="cuda", dtype=torch.long)
    experts = torch.tensor([0, 1, -1], device="cuda", dtype=torch.long)

    y = grouped(x, offsets, experts)
    refs = []
    start = 0
    for group, rows in enumerate(counts):
        host = HostWeight.from_tensor(weight[group], pin_memory=True)
        refs.append(_quantized_linear_reference(x[start : start + rows], host, precision))
        start += rows
    y_ref = torch.cat(refs, dim=0)
    forward_diff = _relative_max_diff(y, y_ref)

    grad_out = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, grad_out)
    dx_refs = []
    start = 0
    for group, rows in enumerate(counts):
        host = HostWeight.from_tensor(weight[group], pin_memory=True)
        dx_refs.append(_quantized_linear_reference(grad_out[start : start + rows].contiguous(), host, precision, transpose=True))
        start += rows
    dx_ref = torch.cat(dx_refs, dim=0)
    dx_diff = _relative_max_diff(dx, dx_ref)

    print(
        f"\n[M1 direct grouped {precision.upper()} quantized error] "
        f"forward_rel_max={forward_diff:.6g}, dx_rel_max={dx_diff:.6g}, "
        f"asym_forward_calls={stats.asym_forward_calls}, asym_dx_calls={stats.asym_dx_calls}"
    )

    assert forward_diff < 3e-2
    assert dx_diff < 3e-2
    assert stats.asym_forward_calls == 1
    assert stats.asym_dx_calls == 1
    assert stats.staged_calls == 0
    assert stats.torch_calls == 0
    assert grouped.host_weight.weight.device.type == "cpu"


def test_lora_composition_gets_grads_and_base_stays_frozen() -> None:
    torch.manual_seed(2)
    use_cuda = _direct_bf16_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    backend = "asym" if use_cuda else "torch"
    m = in_features = out_features = 128 if use_cuda else 8
    rank = 4

    x = torch.randn(m, in_features, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(out_features, in_features, dtype=dtype)
    base = AsymFrozenLinear(weight, backend=backend, pin_memory=use_cuda)
    lora_a = torch.nn.Parameter(torch.randn(rank, in_features, device=device, dtype=dtype) * 0.01)
    lora_b = torch.nn.Parameter(torch.randn(out_features, rank, device=device, dtype=dtype) * 0.01)

    y = base(x) + (x @ lora_a.t() @ lora_b.t())
    loss = y.float().square().mean()
    loss.backward()

    assert base.host_weight.weight.device.type == "cpu"
    assert base.host_weight.weight.grad is None
    assert lora_a.grad is not None
    assert lora_b.grad is not None
    assert x.grad is not None
    assert float(lora_a.grad.float().abs().sum()) > 0.0
    assert float(lora_b.grad.float().abs().sum()) > 0.0


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_lora_composition_quantized_precision_gets_grads_and_base_stays_frozen(precision: str) -> None:
    if not _direct_precision_available(precision):
        pytest.skip(f"direct {precision.upper()} AsymGEMM is not available on this device/build")

    torch.manual_seed(18)
    m = 128
    in_features = 512
    out_features = 128
    rank = 8
    x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    base = AsymFrozenLinear(weight, backend="asym", pin_memory=True, precision=precision)
    lora_a = torch.nn.Parameter(torch.randn(rank, in_features, device="cuda", dtype=torch.float32) * 0.01)
    lora_b = torch.nn.Parameter(torch.randn(out_features, rank, device="cuda", dtype=torch.float32) * 0.01)

    before = base.host_weight.weight.clone()
    y = base(x) + (x.float() @ lora_a.t() @ lora_b.t()).to(torch.bfloat16)
    loss = y.float().square().mean()
    loss.backward()

    assert torch.equal(base.host_weight.weight, before)
    assert base.host_weight.weight.device.type == "cpu"
    assert base.host_weight.weight.grad is None
    assert lora_a.grad is not None and torch.isfinite(lora_a.grad).all()
    assert lora_b.grad is not None and torch.isfinite(lora_b.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(lora_a.grad.float().abs().sum()) > 0.0
    assert float(lora_b.grad.float().abs().sum()) > 0.0


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_lora_quantized_precision_gradients_match_quantized_reference_and_optimizer_only_updates_lora(
    precision: str,
) -> None:
    if not _direct_precision_available(precision):
        pytest.skip(f"direct {precision.upper()} AsymGEMM is not available on this device/build")

    torch.manual_seed(20)
    m = 128
    in_features = 512
    out_features = 128
    rank = 8
    scaling = 2.0
    x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(m, out_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    base = AsymFrozenLinear(weight, backend="asym", pin_memory=True, precision=precision)
    lora_a = torch.nn.Parameter(torch.randn(rank, in_features, device="cuda", dtype=torch.float32) * 0.01)
    lora_b = torch.nn.Parameter(torch.randn(out_features, rank, device="cuda", dtype=torch.float32) * 0.01)
    lora_a_ref = torch.nn.Parameter(lora_a.detach().clone())
    lora_b_ref = torch.nn.Parameter(lora_b.detach().clone())

    y = base(x) + (x.float() @ lora_a.t() @ lora_b.t() * scaling).to(torch.bfloat16)
    y_ref = _quantized_linear_reference(x, base.host_weight, precision).to(torch.bfloat16)
    y_ref = y_ref + (x.float() @ lora_a_ref.t() @ lora_b_ref.t() * scaling).to(torch.bfloat16)
    loss = torch.nn.functional.mse_loss(y.float(), target.float())
    loss_ref = torch.nn.functional.mse_loss(y_ref.float(), target.float())
    loss.backward()
    loss_ref.backward()

    grad_a_diff = _relative_max_diff(lora_a.grad, lora_a_ref.grad)
    grad_b_diff = _relative_max_diff(lora_b.grad, lora_b_ref.grad)
    print(
        f"\n[M1 direct {precision.upper()} LoRA quantized-ref gradients] "
        f"lora_a_rel_max={grad_a_diff:.6g}, lora_b_rel_max={grad_b_diff:.6g}"
    )
    assert grad_a_diff < 3e-2
    assert grad_b_diff < 3e-2

    base_before = base.host_weight.weight.clone()
    lora_a_before = lora_a.detach().clone()
    lora_b_before = lora_b.detach().clone()
    optimizer = torch.optim.SGD([lora_a, lora_b], lr=1e-1)
    optimizer.step()

    assert torch.equal(base.host_weight.weight, base_before)
    assert base.host_weight.weight.grad is None
    assert not torch.equal(lora_a.detach(), lora_a_before)
    assert not torch.equal(lora_b.detach(), lora_b_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory accounting requires CUDA")
def test_cpu_resident_weight_avoids_gpu_hbm_allocation() -> None:
    torch.manual_seed(3)
    weight = torch.randn(2048, 1024, dtype=torch.bfloat16)
    expected_weight_bytes = weight.numel() * weight.element_size()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    host_weight = HostWeight.from_tensor(weight, pin_memory=True)
    torch.cuda.synchronize()
    host_delta = torch.cuda.memory_allocated() - before
    gpu_alloc = measure_gpu_weight_allocation(weight)

    print(
        "\n[M1 CPU-resident memory comparison] "
        f"normal_gpu_weight_hbm_bytes={gpu_alloc}, "
        f"asym_host_weight_cuda_delta_bytes={host_delta}, "
        f"asym_pinned_cpu_bytes={host_weight.pinned_cpu_bytes}, "
        f"expected_weight_bytes={expected_weight_bytes}"
    )

    assert host_weight.weight.device.type == "cpu"
    assert host_delta < 1024 * 1024
    assert gpu_alloc >= expected_weight_bytes
    assert host_weight.weight_nbytes == expected_weight_bytes
    assert host_weight.pinned_cpu_bytes >= expected_weight_bytes


def test_torch_backend_matches_torch_output_loss_and_dx() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(7, 11, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(13, 11, dtype=torch.float32)
    bias = torch.randn(13, dtype=torch.float32)
    host_weight = HostWeight.from_tensor(weight, pin_memory=torch.cuda.is_available())
    stats = AsymExecutionStats()

    y = asym_frozen_linear(x, host_weight, bias=bias, backend="torch", stats=stats)
    loss = (y.square().mean() + y[:, :3].sum() * 0.01)
    dx = torch.autograd.grad(loss, x)[0]

    y_ref = torch.nn.functional.linear(x_ref, weight.to(device=device), bias.to(device=device))
    loss_ref = (y_ref.square().mean() + y_ref[:, :3].sum() * 0.01)
    dx_ref = torch.autograd.grad(loss_ref, x_ref)[0]

    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(loss, loss_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dx, dx_ref, atol=1e-5, rtol=1e-5)
    assert host_weight.weight.grad is None
    assert host_weight.weight.device.type == "cpu"
    assert stats.asym_calls == 0
    assert stats.torch_calls == 2


@pytest.mark.skipif(not _direct_bf16_available(), reason="direct BF16 AsymGEMM requires SM90/SM100")
def test_bf16_asym_backend_raises_when_direct_kernel_is_unavailable_cuda() -> None:
    torch.manual_seed(5)
    x = torch.randn(17, 19, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(23, 19, dtype=torch.bfloat16)
    bias = torch.randn(23, dtype=torch.bfloat16)
    host_weight = HostWeight.from_tensor(weight, pin_memory=True)
    stats = AsymExecutionStats()

    ok, reason = can_use_direct_bf16(x, host_weight)
    assert not ok and reason == "requires_8_aligned_nk"

    with pytest.raises(RuntimeError, match="direct BF16 AsymGEMM is unavailable: requires_8_aligned_nk"):
        asym_frozen_linear(x, host_weight, bias=bias, backend="asym", stats=stats)

    assert stats.asym_calls == 0
    assert stats.torch_calls == 0
    assert stats.fallback_reasons == {"forward:requires_8_aligned_nk": 1}
    assert host_weight.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for module.to('cuda') residency check")
def test_module_to_cuda_does_not_move_host_weight_or_register_it() -> None:
    torch.manual_seed(6)
    weight = torch.randn(9, 7, dtype=torch.float32)
    module = AsymFrozenLinear(7, 9, weight, backend="torch")
    ptr_before = module.host_weight.weight.data_ptr()

    module = module.to("cuda")

    assert module.host_weight.weight.device.type == "cpu"
    assert module.host_weight.weight.data_ptr() == ptr_before
    assert module.weight.device.type == "cpu"
    assert list(module.parameters()) == []
    assert "host_weight" not in dict(module.named_buffers())
    with pytest.raises(RuntimeError, match="CPU-resident"):
        module.host_weight.to("cuda")
    with pytest.raises(RuntimeError, match="CPU-resident"):
        module.host_weight.cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for peak HBM accounting")
def test_cpu_resident_forward_excludes_weight_bytes_from_peak_cuda_allocation() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch = 4
    in_features = 2048
    out_features = 2048

    x = torch.randn(batch, in_features, device=device, dtype=torch.float32)
    weight_cpu = torch.randn(out_features, in_features, dtype=torch.float32)
    host_weight = HostWeight.from_tensor(weight_cpu, pin_memory=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    y_host = asym_frozen_linear(x, host_weight, backend="torch")
    torch.cuda.synchronize()
    host_peak = torch.cuda.max_memory_allocated()
    del y_host

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    weight_gpu = weight_cpu.to(device=device)
    y_gpu = torch.nn.functional.linear(x, weight_gpu)
    torch.cuda.synchronize()
    gpu_peak = torch.cuda.max_memory_allocated()
    del y_gpu, weight_gpu

    assert host_weight.weight.device.type == "cpu"
    assert gpu_peak - host_peak >= int(host_weight.weight_nbytes * 0.8)


def test_backend_names_are_stable() -> None:
    assert VALID_BACKENDS == ("asym", "torch")
