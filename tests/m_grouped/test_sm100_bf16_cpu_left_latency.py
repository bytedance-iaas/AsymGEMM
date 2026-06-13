from __future__ import annotations

import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major


CPU_LEFT_BINDING = "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous"

pytestmark = pytest.mark.skipif(
    os.environ.get("ASYM_RUN_PERF_TESTS") != "1"
    or not torch.cuda.is_available()
    or get_arch_major() != 10
    or not hasattr(asym_gemm, CPU_LEFT_BINDING),
    reason="set ASYM_RUN_PERF_TESTS=1 on SM100 with CPU-left binding",
)


def _pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    pinned = tensor.detach().cpu().contiguous().pin_memory()
    assert pinned.is_pinned()
    return pinned


def _metadata(rows: int):
    return (
        torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        torch.tensor([0, -1], device="cuda", dtype=torch.int32),
        2,
    )


def _measure_ms(fn, *, warmup: int, iters: int) -> list[float]:
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _case(s: int, k: int, *, n: int | None = None, warmup: int = 10, iters: int = 30):
    torch.manual_seed(0)
    n = s if n is None else n
    a_cuda = torch.randn((s, k), device="cuda", dtype=torch.bfloat16)
    b_cuda = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16)
    a_cpu = _pin_cpu(a_cuda)
    b_cpu = _pin_cpu(b_cuda)
    offsets, experts, list_size = _metadata(s)
    left_out = torch.empty((s, n), device="cuda", dtype=torch.bfloat16)
    right_out = torch.empty_like(left_out)

    def left() -> None:
        getattr(asym_gemm, CPU_LEFT_BINDING)(a_cpu, b_cuda, left_out, offsets, experts, list_size, "nk")

    def right() -> None:
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a_cuda, b_cpu, right_out, offsets, experts, list_size, "nk"
        )

    # Touch both pinned operands before timing so the first measured iteration
    # is not dominated by page placement.
    _ = float(a_cpu[0, 0].item()) + float(b_cpu[0, 0, 0].item())
    left()
    right()
    torch.cuda.synchronize()

    torch_ref = a_cuda.float().matmul(b_cuda[0].float().t()).to(torch.bfloat16)
    diff_torch = calc_diff(left_out, torch_ref)
    diff_right = calc_diff(left_out, right_out)

    left_ms = _measure_ms(left, warmup=warmup, iters=iters)
    right_ms = _measure_ms(right, warmup=warmup, iters=iters)
    left_median = float(statistics.median(left_ms))
    right_median = float(statistics.median(right_ms))
    ratio = left_median / right_median if right_median > 0.0 else float("inf")
    return left_median, right_median, ratio, diff_torch, diff_right


@pytest.mark.parametrize(("s", "k"), [(128, 512), (256, 1024), (512, 4096)])
def test_cpu_left_square_latency_close_to_cpu_right(s: int, k: int) -> None:
    left_ms, right_ms, ratio, diff_torch, diff_right = _case(s, k)
    print(
        f"SM100 BF16 CPU-left square S={s} K={k}: "
        f"left_us={left_ms * 1000:.2f} right_us={right_ms * 1000:.2f} "
        f"ratio={ratio:.3f} diff_torch={diff_torch:.5e} diff_right={diff_right:.5e}"
    )

    assert diff_torch < 1e-3
    assert diff_right < 1e-3
    # CUDA-event timings include launch-path gaps that NCU does not attribute to
    # kernel duration. Keep this opt-in test as a broad regression smoke check;
    # use NCU for kernel-level parity decisions.
    max_ratio = 1.60 if s <= 256 else 1.25
    assert 0.85 <= ratio <= max_ratio


@pytest.mark.parametrize("rank", [8, 16, 64, 128])
def test_cpu_left_lora_rank_latency_report_only(rank: int) -> None:
    left_ms, right_ms, ratio, diff_torch, diff_right = _case(256, 1024, n=rank)
    print(
        f"SM100 BF16 CPU-left LoRA-rank rank={rank}: "
        f"left_us={left_ms * 1000:.2f} right_us={right_ms * 1000:.2f} "
        f"ratio={ratio:.3f} diff_torch={diff_torch:.5e} diff_right={diff_right:.5e}"
    )

    assert diff_torch < 1e-3
    assert diff_right < 1e-3


def test_cpu_left_profile_script_prints_diff_and_ratio(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "lora" / "profile_cpu_left_bf16_sm100.py"
    result_json = tmp_path / "cpu_left_profile.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--square-cases",
            "128x512",
            "--lora-ranks",
            "8",
            "--warmup",
            "1",
            "--iters",
            "3",
            "--max-ratio",
            "10.0",
            "--json",
            str(result_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ratio" in result.stdout
    assert "diff_torch" in result.stdout
    assert result_json.is_file()
