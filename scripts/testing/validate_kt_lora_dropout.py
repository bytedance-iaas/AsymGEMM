#!/usr/bin/env python3
"""Validate KT SFT LoRA dropout on tiny deterministic MoE cases."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
KT_KERNEL_DIR = REPO_ROOT.parent / "ktransformers" / "kt-kernel"
if str(KT_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(KT_KERNEL_DIR))

from kt_kernel import kt_kernel_ext  # noqa: E402
from kt_kernel.experts import KTMoEWrapper  # noqa: E402
from kt_kernel.sft.dropout import (  # noqa: E402
    KT_LORA_DROPOUT_DOWN,
    KT_LORA_DROPOUT_GATE,
    KT_LORA_DROPOUT_UP,
    counter_dropout_apply,
)
from kt_kernel.sft.torch_backend import TorchBF16SFTMoEWrapper  # noqa: E402
import kt_kernel.sft.base as kt_base  # noqa: E402
import kt_kernel.sft.torch_backend as kt_torch_backend  # noqa: E402


TORCH_ATOL = 3e-2
TORCH_RTOL = 3e-2
ARM_ATOL = 8e-2
ARM_RTOL = 8e-2


@dataclass
class Case:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    lora: dict[str, torch.Tensor]
    hidden: torch.Tensor
    expert_ids: torch.Tensor
    weights: torch.Tensor
    grad_output: torch.Tensor
    lora_alpha: float


def _max_rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().abs().clamp_min(1e-6)
    return float(((actual.float() - expected.float()).abs() / denom).max().item())


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    return float(diff.max().item()), _max_rel(actual, expected)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def make_case() -> Case:
    torch.manual_seed(20260606)
    num_experts = 4
    top_k = 2
    hidden_size = 8
    intermediate_size = 12
    rank = 3
    qlen = 7
    gate_proj = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16) * 0.2
    up_proj = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16) * 0.2
    down_proj = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16) * 0.2
    lora = {
        "gate_lora_a": torch.randn(num_experts, rank, hidden_size, dtype=torch.bfloat16) * 0.03,
        "gate_lora_b": torch.randn(num_experts, intermediate_size, rank, dtype=torch.bfloat16) * 0.03,
        "up_lora_a": torch.randn(num_experts, rank, hidden_size, dtype=torch.bfloat16) * 0.03,
        "up_lora_b": torch.randn(num_experts, intermediate_size, rank, dtype=torch.bfloat16) * 0.03,
        "down_lora_a": torch.randn(num_experts, rank, intermediate_size, dtype=torch.bfloat16) * 0.03,
        "down_lora_b": torch.randn(num_experts, hidden_size, rank, dtype=torch.bfloat16) * 0.03,
    }
    hidden = torch.randn(qlen, hidden_size, dtype=torch.bfloat16)
    expert_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0]], dtype=torch.long)
    weights = torch.softmax(torch.randn(qlen, top_k), dim=-1)
    grad_output = torch.randn(qlen, hidden_size, dtype=torch.bfloat16)
    return Case(gate_proj, up_proj, down_proj, lora, hidden, expert_ids, weights, grad_output, lora_alpha=6.0)


def reference_forward(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    lora: dict[str, torch.Tensor],
    scaling: float,
    dropout_p: float,
    dropout_seed: int,
    layer_idx: int,
    training: bool,
) -> torch.Tensor:
    hidden = hidden.to(torch.float32)
    weights = weights.to(torch.float32)
    effective_p = dropout_p if training else 0.0
    out = hidden.new_zeros((hidden.shape[0], down_proj.shape[1]))
    for token in range(hidden.shape[0]):
        x = hidden[token : token + 1]
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route])
            token_ids = torch.tensor([token], dtype=torch.long)
            route_slots = torch.tensor([route], dtype=torch.long)
            gate_x = counter_dropout_apply(
                x,
                seed=dropout_seed,
                layer_idx=layer_idx,
                projection_id=KT_LORA_DROPOUT_GATE,
                expert_id=expert,
                token_ids=token_ids,
                route_slots=route_slots,
                p=effective_p,
            )[0]
            up_x = counter_dropout_apply(
                x,
                seed=dropout_seed,
                layer_idx=layer_idx,
                projection_id=KT_LORA_DROPOUT_UP,
                expert_id=expert,
                token_ids=token_ids,
                route_slots=route_slots,
                p=effective_p,
            )[0]
            x_vec = x[0]
            gate = gate_proj[expert].to(torch.float32).matmul(x_vec)
            gate = gate + lora["gate_lora_b"][expert].to(torch.float32).matmul(
                lora["gate_lora_a"][expert].to(torch.float32).matmul(gate_x)
            ) * scaling
            up = up_proj[expert].to(torch.float32).matmul(x_vec)
            up = up + lora["up_lora_b"][expert].to(torch.float32).matmul(
                lora["up_lora_a"][expert].to(torch.float32).matmul(up_x)
            ) * scaling
            act = F.silu(gate) * up
            down_act = counter_dropout_apply(
                act.unsqueeze(0),
                seed=dropout_seed,
                layer_idx=layer_idx,
                projection_id=KT_LORA_DROPOUT_DOWN,
                expert_id=expert,
                token_ids=token_ids,
                route_slots=route_slots,
                p=effective_p,
            )[0]
            down = down_proj[expert].to(torch.float32).matmul(act)
            down = down + lora["down_lora_b"][expert].to(torch.float32).matmul(
                lora["down_lora_a"][expert].to(torch.float32).matmul(down_act)
            ) * scaling
            out[token] = out[token] + weights[token, route] * down
    return out


def make_backend(backend: str, case: Case, dropout: float):
    rank = case.lora["gate_lora_a"].shape[1]
    kwargs = dict(
        layer_idx=2,
        num_experts=case.gate_proj.shape[0],
        num_experts_per_tok=case.expert_ids.shape[1],
        hidden_size=case.hidden.shape[1],
        moe_intermediate_size=case.gate_proj.shape[1],
        num_gpu_experts=0,
        cpuinfer_threads=1,
        threadpool_count=1,
        weight_path="",
        chunked_prefill_size=32,
        lora_rank=rank,
        lora_alpha=case.lora_alpha,
        lora_dropout=dropout,
        max_cache_depth=2,
    )
    if backend == "kt_torchbf16":
        wrapper = TorchBF16SFTMoEWrapper(**kwargs)
    elif backend == "kt_armbf16":
        if getattr(kt_kernel_ext, "__python_fallback__", False) or not hasattr(kt_kernel_ext.moe, "ARMBF16_SFT_MOE"):
            raise RuntimeError("kt_armbf16 requires native ARMBF16_SFT_MOE in kt_kernel_ext on an aarch64 host")
        wrapper = KTMoEWrapper(
            gpu_experts_mask=None,
            method="ARMBF16_SFT",
            mode="sft",
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported backend {backend!r}")
    wrapper.load_weights_from_tensors(case.gate_proj, case.up_proj, case.down_proj, torch.arange(case.gate_proj.shape[0]))
    grad_buffers = {f"grad_{name}": torch.zeros_like(tensor, dtype=torch.bfloat16) for name, tensor in case.lora.items()}
    wrapper.init_lora_weights(**case.lora, **grad_buffers)
    return wrapper, grad_buffers


def run_backend(backend: str, case: Case, dropout: float, checkpoint: str, training: bool, seed: int) -> dict[str, Any]:
    old_torch_seed = kt_torch_backend.next_lora_dropout_seed
    old_base_seed = kt_base.next_lora_dropout_seed
    kt_torch_backend.next_lora_dropout_seed = lambda enabled: seed if enabled else 0
    kt_base.next_lora_dropout_seed = lambda enabled: seed if enabled else 0
    try:
        wrapper, grad_buffers = make_backend(backend, case, dropout)
        if checkpoint == "on":
            first = wrapper.forward(
                case.hidden,
                case.expert_ids,
                case.weights,
                save_for_backward=False,
                output_device=torch.device("cpu"),
                training=training,
            )
            output = wrapper.forward(
                case.hidden,
                case.expert_ids,
                case.weights,
                save_for_backward=training,
                output_device=torch.device("cpu"),
                training=training,
            )
            torch.testing.assert_close(first.float(), output.float(), atol=0.0, rtol=0.0)
        else:
            output = wrapper.forward(
                case.hidden,
                case.expert_ids,
                case.weights,
                save_for_backward=training,
                output_device=torch.device("cpu"),
                training=training,
            )

        result: dict[str, Any] = {"output": output}
        if training:
            grad_input, grad_weights = wrapper.backward(case.grad_output, output_device=torch.device("cpu"))
            result.update({"grad_input": grad_input, "grad_weights": grad_weights})
            for name in case.lora:
                result[f"grad_{name}"] = grad_buffers[f"grad_{name}"]
        return result
    finally:
        kt_torch_backend.next_lora_dropout_seed = old_torch_seed
        kt_base.next_lora_dropout_seed = old_base_seed


def run_reference(case: Case, dropout: float, training: bool, seed: int) -> dict[str, torch.Tensor]:
    hidden = case.hidden.to(torch.float32).detach().clone().requires_grad_(training)
    weights = case.weights.to(torch.float32).detach().clone().requires_grad_(training)
    lora = {name: tensor.to(torch.float32).detach().clone().requires_grad_(training) for name, tensor in case.lora.items()}
    scaling = case.lora_alpha / case.lora["gate_lora_a"].shape[1]
    output = reference_forward(
        hidden,
        case.expert_ids,
        weights,
        case.gate_proj,
        case.up_proj,
        case.down_proj,
        lora,
        scaling,
        dropout,
        seed,
        layer_idx=2,
        training=training,
    )
    result = {"output": output}
    if training:
        output.backward(case.grad_output.to(torch.float32))
        result["grad_input"] = hidden.grad
        result["grad_weights"] = weights.grad
        for name, tensor in lora.items():
            result[f"grad_{name}"] = tensor.grad
    return result


def compare(label: str, actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], atol: float, rtol: float) -> None:
    keys = sorted(expected)
    failed = []
    for key in keys:
        _assert_finite(f"{label}:{key}:actual", actual[key])
        _assert_finite(f"{label}:{key}:expected", expected[key])
        max_abs, max_rel = _metrics(actual[key], expected[key])
        print(f"{label:18s} {key:18s} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        if max_abs > atol and max_rel > rtol:
            failed.append(f"{key} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
    if failed:
        raise AssertionError(f"{label} exceeded tolerances: " + "; ".join(failed))


def profile_backend(backend: str, case: Case, dropout: float, checkpoint: str, training: bool, seed: int, iters: int) -> None:
    start = time.perf_counter()
    for _ in range(iters):
        run_backend(backend, case, dropout, checkpoint, training, seed)
    elapsed = time.perf_counter() - start
    print(f"profile backend={backend} dropout={dropout} checkpoint={checkpoint} training={training} iters={iters} avg_s={elapsed / iters:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["kt_torchbf16", "kt_armbf16"], required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--checkpoint", choices=["off", "on"], default="off")
    parser.add_argument("--training", choices=["true", "false"], default="true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=5)
    args = parser.parse_args()

    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must satisfy 0 <= p < 1")
    training = args.training == "true"
    case = make_case()

    actual = run_backend(args.backend, case, args.dropout, args.checkpoint, training, args.seed)
    if args.backend == "kt_torchbf16":
        expected = run_reference(case, args.dropout, training, args.seed)
        compare(args.backend, actual, expected, TORCH_ATOL, TORCH_RTOL)
    else:
        expected = run_backend("kt_torchbf16", case, args.dropout, args.checkpoint, training, args.seed)
        compare(args.backend, actual, expected, ARM_ATOL, ARM_RTOL)
        if args.dropout == 0.0:
            manual = run_reference(case, 0.0, training, args.seed)
            compare(f"{args.backend}:p0", actual, manual, ARM_ATOL, ARM_RTOL)

    if args.profile:
        profile_backend(args.backend, case, args.dropout, args.checkpoint, training, args.seed, args.profile_iters)


if __name__ == "__main__":
    main()
