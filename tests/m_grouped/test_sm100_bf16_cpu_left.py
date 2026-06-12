import random

import pytest
import torch

import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major


CPU_LEFT_BINDING = "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_arch_major() != 10,
    reason="SM100/GB200 required",
)


def _require_cpu_left_binding() -> None:
    if not hasattr(asym_gemm, CPU_LEFT_BINDING):
        pytest.skip("SM100 BF16 CPU-left grouped kernel is not exported")


def _seed() -> None:
    random.seed(0)
    torch.manual_seed(0)


def _pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    pinned = tensor.detach().cpu().pin_memory()
    assert pinned.device.type == "cpu" and pinned.is_pinned()
    return pinned


def _metadata(lengths: list[int], experts: list[int] | None = None):
    if experts is None:
        experts = list(range(len(lengths)))
    assert len(experts) == len(lengths)
    pair_offsets: list[int] = []
    cumulative = [0]
    start = 0
    for rows in lengths:
        pair_offsets.extend([start, start + rows])
        start += rows
        cumulative.append(start)
    return (
        torch.tensor(pair_offsets, dtype=torch.int32, device="cuda"),
        torch.tensor(cumulative, dtype=torch.int32, device="cuda"),
        torch.tensor([*experts, -1], dtype=torch.int32, device="cuda"),
        len(experts) + 1,
    )


def _reference(a_cpu: torch.Tensor, b_cuda: torch.Tensor, cumulative: torch.Tensor, experts: torch.Tensor, out_dtype):
    offsets = cumulative.detach().cpu().tolist()
    expert_ids = experts.detach().cpu().tolist()
    out = torch.empty((int(a_cpu.shape[0]), int(b_cuda.shape[1])), device="cuda", dtype=out_dtype)
    for group, expert in enumerate(expert_ids[:-1]):
        start = int(offsets[group])
        end = int(offsets[group + 1])
        if end <= start:
            continue
        rows = a_cpu[start:end].to(device="cuda").float().matmul(b_cuda[int(expert)].float().t())
        out[start:end] = rows.to(dtype=out_dtype)
    return out


def _run_cpu_left(a_cpu, b_cuda, pair_offsets, experts, list_size, out_dtype=torch.bfloat16):
    d = torch.empty((int(a_cpu.shape[0]), int(b_cuda.shape[1])), device="cuda", dtype=out_dtype)
    getattr(asym_gemm, CPU_LEFT_BINDING)(a_cpu, b_cuda, d, pair_offsets, experts, list_size, "nk")
    torch.cuda.synchronize()
    return d


@pytest.mark.parametrize(
    ("lengths", "experts", "n", "k", "out_dtype"),
    [
        ([128], [0], 8, 512, torch.bfloat16),
        ([256], [0], 16, 1024, torch.bfloat16),
        ([128, 0, 192, 64], [0, 1, 2, 3], 16, 512, torch.bfloat16),
        ([64, 96, 32, 128], [2, 0, 2, 1], 64, 1024, torch.bfloat16),
        ([128, 128], [0, 1], 128, 4096, torch.bfloat16),
        ([64, 64], [0, 1], 64, 1024, torch.float32),
    ],
)
def test_cpu_left_matches_torch_sm100_bf16(lengths, experts, n, k, out_dtype) -> None:
    _require_cpu_left_binding()
    _seed()
    m = sum(lengths)
    num_experts = max(experts) + 1
    a_cuda = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    a_cpu = _pin_cpu(a_cuda)
    b_cuda = torch.randn((num_experts, n, k), device="cuda", dtype=torch.bfloat16)
    pair_offsets, cumulative, experts_t, list_size = _metadata(lengths, experts)

    out = _run_cpu_left(a_cpu, b_cuda, pair_offsets, experts_t, list_size, out_dtype=out_dtype)
    ref = _reference(a_cpu, b_cuda, cumulative, experts_t, out_dtype)

    diff = calc_diff(out, ref)
    assert diff < 1e-3, f"CPU-left torch diff={diff:.5e} lengths={lengths} n={n} k={k}"


def test_cpu_left_tensor_list_size_matches_torch_with_repeated_routes() -> None:
    _require_cpu_left_binding()
    _seed()
    lengths = [64, 96, 32, 128]
    experts = [2, 0, 2, 1]
    m = sum(lengths)
    n = 64
    k = 1024
    a_cuda = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    a_cpu = _pin_cpu(a_cuda)
    b_cuda = torch.randn((3, n, k), device="cuda", dtype=torch.bfloat16)
    pair_offsets, cumulative, experts_t, list_size = _metadata(lengths, experts)
    list_size_t = torch.tensor([list_size], device="cuda", dtype=torch.int32)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    getattr(asym_gemm, CPU_LEFT_BINDING)(a_cpu, b_cuda, out, pair_offsets, experts_t, list_size_t, "nk")
    torch.cuda.synchronize()
    ref = _reference(a_cpu, b_cuda, cumulative, experts_t, torch.bfloat16)

    diff = calc_diff(out, ref)
    assert diff < 1e-3, f"CPU-left tensor list_size torch diff={diff:.5e}"


def test_cpu_left_zero_active_rows_is_noop() -> None:
    _require_cpu_left_binding()
    _seed()
    a_cpu = _pin_cpu(torch.empty((0, 512), dtype=torch.bfloat16))
    b_cuda = torch.randn((3, 16, 512), device="cuda", dtype=torch.bfloat16)
    pair_offsets, _cumulative, experts_t, list_size = _metadata([0, 0, 0])

    out = _run_cpu_left(a_cpu, b_cuda, pair_offsets, experts_t, list_size)
    assert tuple(out.shape) == (0, 16)


def test_cpu_left_square_matches_cpu_right_asym_and_torch() -> None:
    _require_cpu_left_binding()
    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        pytest.skip("BF16 CPU-right grouped kernel is not exported")

    _seed()
    s, k = 256, 1024
    a_cuda = torch.randn((s, k), device="cuda", dtype=torch.bfloat16)
    b_cuda = torch.randn((1, s, k), device="cuda", dtype=torch.bfloat16)
    a_cpu = _pin_cpu(a_cuda)
    b_cpu = _pin_cpu(b_cuda)
    pair_offsets, _cumulative, experts_t, list_size = _metadata([s], [0])

    left = _run_cpu_left(a_cpu, b_cuda, pair_offsets, experts_t, list_size)
    right = torch.empty_like(left)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a_cuda, b_cpu, right, pair_offsets, experts_t, list_size, "nk"
    )
    torch.cuda.synchronize()
    torch_ref = a_cuda.float().matmul(b_cuda[0].float().t()).to(torch.bfloat16)

    diff_left = calc_diff(left, torch_ref)
    diff_right = calc_diff(right, torch_ref)
    diff_left_right = calc_diff(left, right)
    assert diff_left < 1e-3, f"CPU-left torch diff={diff_left:.5e}"
    assert diff_right < 1e-3, f"CPU-right torch diff={diff_right:.5e}"
    assert diff_left_right < 1e-3, f"CPU-left/right diff={diff_left_right:.5e}"


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda a, b, d, o, e, l: (a.to("cuda"), b, d, o, e, l), "input_not_cpu"),
        (lambda a, b, d, o, e, l: (a.detach().clone(), b, d, o, e, l), "input_not_pinned"),
        (lambda a, b, d, o, e, l: (a, b.cpu(), d, o, e, l), "weight_not_cuda"),
        (lambda a, b, d, o, e, l: (a, b, d.cpu(), o, e, l), "output_not_cuda"),
        (lambda a, b, d, o, e, l: (a.float(), b, d, o, e, l), "requires_bf16"),
        (lambda a, b, d, o, e, l: (a, b.transpose(-1, -2), d, o, e, l), "requires_contiguous"),
        (
            lambda a, b, d, o, e, l: (
                a,
                torch.randn((1, 10, a.shape[1]), device="cuda", dtype=torch.bfloat16),
                torch.empty((a.shape[0], 10), device="cuda", dtype=torch.bfloat16),
                o,
                e,
                l,
            ),
            "requires_8_aligned_nk",
        ),
    ],
)
def test_cpu_left_guard_failures_are_named(mutator, reason) -> None:
    _require_cpu_left_binding()
    _seed()
    a_cpu = _pin_cpu(torch.randn((128, 512), device="cuda", dtype=torch.bfloat16))
    b_cuda = torch.randn((1, 16, 512), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((128, 16), device="cuda", dtype=torch.bfloat16)
    pair_offsets, _cumulative, experts_t, list_size = _metadata([128], [0])

    args = mutator(a_cpu, b_cuda, d, pair_offsets, experts_t, list_size)
    with pytest.raises(RuntimeError, match=reason):
        getattr(asym_gemm, CPU_LEFT_BINDING)(*args, "nk")
