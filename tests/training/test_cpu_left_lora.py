from __future__ import annotations

import importlib

import pytest
import torch

import asym_gemm
from asym_gemm.training import AsymExecutionStats


cpu_left_impl = importlib.import_module("asym_gemm.training.cpu_left")
lora_impl = importlib.import_module("asym_gemm.training.lora")


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinning required")


def _pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    pinned = tensor.detach().cpu().contiguous().pin_memory()
    if not pinned.is_pinned():
        pytest.skip("pin_memory unavailable")
    return pinned


def _metadata(lengths: list[int], experts: list[int]):
    offsets = [0]
    for rows in lengths:
        offsets.append(offsets[-1] + int(rows))
    return (
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        torch.tensor([*experts, -1], device="cuda", dtype=torch.int32),
    )


def _reference(x_cpu: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
    offsets_cpu = offsets.detach().cpu().tolist()
    experts_cpu = experts.detach().cpu().tolist()
    rows: list[torch.Tensor] = []
    for group, expert in enumerate(experts_cpu[:-1]):
        start = int(offsets_cpu[group])
        end = int(offsets_cpu[group + 1])
        if end <= start:
            continue
        rows.append(x_cpu[start:end].to(device="cuda").float().matmul(weight[int(expert)].float().t()))
    if rows:
        return torch.cat(rows, dim=0).to(torch.bfloat16)
    return torch.empty((0, int(weight.shape[1])), device="cuda", dtype=torch.bfloat16)


def _install_reference_binding(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, object]] | None = None) -> None:
    monkeypatch.setattr(cpu_left_impl, "_arch_major", lambda device: 10)

    def fake_binding(
        a_cpu: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        pair_offsets: torch.Tensor,
        experts: torch.Tensor,
        list_size: int,
        compiled_dims: str,
    ) -> None:
        if calls is not None:
            calls.append(
                {
                    "a_device": a_cpu.device.type,
                    "a_data_ptr": a_cpu.data_ptr(),
                    "weight_data_ptr": weight.data_ptr(),
                    "pair_offsets_shape": tuple(pair_offsets.shape),
                    "experts_shape": tuple(experts.shape),
                    "list_size": int(list_size),
                    "compiled_dims": compiled_dims,
                }
            )
        pair_offsets_cpu = pair_offsets.detach().cpu().tolist()
        experts_cpu = experts.detach().cpu().tolist()
        out.zero_()
        for group in range(int(list_size) - 1):
            start = int(pair_offsets_cpu[2 * group])
            end = int(pair_offsets_cpu[2 * group + 1])
            if end <= start:
                continue
            expert = int(experts_cpu[group])
            rows = a_cpu[start:end].to(device=out.device).float().matmul(weight[expert].float().t())
            out[start:end].copy_(rows.to(dtype=out.dtype))

    monkeypatch.setattr(asym_gemm, cpu_left_impl.CPU_LEFT_BF16_BINDING, fake_binding, raising=False)


def test_grouped_expert_lora_cpu_left_matches_cuda_grouped_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    _install_reference_binding(monkeypatch)

    lengths = [64, 0, 96, 32]
    experts = [2, 0, 2, 1]
    m = sum(lengths)
    k = 128
    n = 16
    x_cuda = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    x_cpu = _pin_cpu(x_cuda)
    weight = torch.randn((3, n, k), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata(lengths, experts)

    out = cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t)
    ref = _reference(x_cpu, weight, offsets, experts_t)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


def test_grouped_expert_lora_cpu_left_dense_metadata_avoids_weight_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(1)
    calls: list[dict[str, object]] = []
    _install_reference_binding(monkeypatch, calls)

    lengths = [64, 0, 96, 32]
    experts = [0, 1, 2, 3]
    m = sum(lengths)
    k = 128
    n = 16
    x_cpu = _pin_cpu(torch.randn((m, k), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((4, n, k), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata(lengths, experts)
    metadata = lora_impl.prepare_grouped_lora_metadata(offsets, experts_t, dense_experts=True)

    out = cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t, metadata=metadata)
    ref = _reference(x_cpu, weight, offsets, experts_t)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
    assert calls and calls[0]["weight_data_ptr"] == weight.data_ptr()


def test_grouped_expert_lora_cpu_left_does_not_stage_full_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(2)
    calls: list[dict[str, object]] = []
    _install_reference_binding(monkeypatch, calls)

    m = 192
    k = 128
    n = 16
    x_cpu = _pin_cpu(torch.randn((m, k), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((2, n, k), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata([96, 96], [0, 1])
    cuda_empty_shapes: list[tuple[int, ...]] = []
    real_empty = torch.empty

    def counted_empty(*args, **kwargs):
        device_arg = kwargs.get("device", None)
        device = torch.device(device_arg) if device_arg is not None else torch.device("cpu")
        if device.type == "cuda":
            shape = args[0] if args else kwargs.get("size")
            cuda_empty_shapes.append(tuple(shape))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(cpu_left_impl.torch, "empty", counted_empty)

    out = cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t)

    assert tuple(out.shape) == (m, n)
    assert calls and calls[0]["a_device"] == "cpu"
    assert calls[0]["a_data_ptr"] == x_cpu.data_ptr()
    assert (m, k) not in cuda_empty_shapes


def test_grouped_expert_lora_cpu_left_empty_groups_match_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(3)
    _install_reference_binding(monkeypatch)

    lengths = [0, 64, 0, 64]
    experts = [0, 1, 2, 1]
    m = sum(lengths)
    k = 128
    n = 16
    x_cpu = _pin_cpu(torch.randn((m, k), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((3, n, k), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata(lengths, experts)

    out = cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t)
    ref = _reference(x_cpu, weight, offsets, experts_t)

    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda x, w, o, e: (x.to("cuda"), w, o, e), "input_not_cpu"),
        (lambda x, w, o, e: (x.detach().clone(), w, o, e), "input_not_pinned"),
        (lambda x, w, o, e: (x, w.cpu(), o, e), "weight_not_cuda"),
        (lambda x, w, o, e: (x.float(), w, o, e), "requires_bf16"),
        (lambda x, w, o, e: (x[:, :64], w, o, e), "requires_contiguous"),
        (
            lambda x, w, o, e: (x, torch.randn((1, 10, x.shape[1]), device="cuda", dtype=torch.bfloat16), o, e),
            "requires_8_aligned_nk",
        ),
    ],
)
def test_grouped_expert_lora_cpu_left_guard_reasons(
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    reason: str,
) -> None:
    torch.manual_seed(4)
    _install_reference_binding(monkeypatch)

    x_cpu = _pin_cpu(torch.randn((128, 128), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata([128], [0])
    args = mutator(x_cpu, weight, offsets, experts_t)

    with pytest.raises(RuntimeError, match=reason):
        cpu_left_impl.grouped_expert_lora_cpu_left(*args)


def test_grouped_expert_lora_cpu_left_missing_binding_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(5)
    monkeypatch.setattr(cpu_left_impl, "_arch_major", lambda device: 10)
    monkeypatch.delattr(asym_gemm, cpu_left_impl.CPU_LEFT_BF16_BINDING, raising=False)

    x_cpu = _pin_cpu(torch.randn((128, 128), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata([128], [0])

    with pytest.raises(RuntimeError, match="missing_sm100_cpu_left_bf16_binding"):
        cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t)


def test_grouped_expert_lora_cpu_left_invalid_output_dtype_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(6)
    _install_reference_binding(monkeypatch)

    x_cpu = _pin_cpu(torch.randn((128, 128), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata([128], [0])

    with pytest.raises(RuntimeError, match="requires_bf16_or_fp32_output"):
        cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t, output_dtype=torch.float16)


def test_grouped_expert_lora_cpu_left_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    _install_reference_binding(monkeypatch)

    x_cpu = _pin_cpu(torch.randn((128, 128), device="cuda", dtype=torch.bfloat16))
    weight = torch.randn((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    offsets, experts_t = _metadata([128], [0])
    stats = AsymExecutionStats()

    cpu_left_impl.grouped_expert_lora_cpu_left(x_cpu, weight, offsets, experts_t, stats=stats)

    assert stats.cpu_left_lora_a_calls == 1
