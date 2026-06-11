#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _case_config(name: str) -> dict[str, int | list[int]]:
    cases: dict[str, dict[str, int | list[int]]] = {
        "tiny": {"E": 3, "H": 16, "I": 10, "R": 4, "counts": [2, 0, 3], "P": 4, "Q": 2, "BM": 2, "BK": 8, "G": 2},
        "one_group": {"E": 2, "H": 256, "I": 512, "R": 8, "counts": [8, 0], "P": 32, "Q": 4, "BM": 32, "BK": 128, "G": 2},
        "ragged_groups": {"E": 4, "H": 384, "I": 768, "R": 16, "counts": [3, 0, 17, 19], "P": 32, "Q": 4, "BM": 32, "BK": 128, "G": 4},
        "partial_window": {"E": 3, "H": 512, "I": 650, "R": 8, "counts": [7, 0, 12], "P": 32, "Q": 8, "BM": 32, "BK": 128, "G": 3},
        "qwen_shape_smallM": {"E": 8, "H": 2048, "I": 768, "R": 64, "counts": [64] * 8, "P": 32, "Q": 8, "BM": 64, "BK": 512, "G": 8},
    }
    if name not in cases:
        raise ValueError(f"unknown case {name!r}; choices={tuple(cases)}")
    return cases[name]


def _build_offsets_experts(counts: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    offsets = [0]
    active_experts: list[int] = []
    for expert, count in enumerate(counts):
        if int(count) <= 0:
            continue
        active_experts.append(expert)
        offsets.append(offsets[-1] + int(count))
    return (
        torch.tensor(offsets, device=device, dtype=torch.int32),
        torch.tensor([*active_experts, -1], device=device, dtype=torch.int32),
        offsets,
        active_experts,
    )


def _reference(
    x: torch.Tensor,
    dact: torch.Tensor,
    dS_down: torch.Tensor | None,
    down_mask_bool: torch.Tensor | None,
    down_dropout_p: float,
    gate_low_rank: torch.Tensor,
    up_low_rank: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_b: torch.Tensor,
    gate_up_weight_cpu: torch.Tensor,
    offsets: list[int],
    experts: list[int],
    lora_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    device = x.device
    m, h = x.shape
    i = dact.shape[1]
    w_gate = gate_up_weight_cpu[:, :i, :].to(device=device, dtype=torch.float32)
    w_up = gate_up_weight_cpu[:, i:, :].to(device=device, dtype=torch.float32)
    gate = torch.empty((m, i), device=device, dtype=torch.float32)
    up = torch.empty((m, i), device=device, dtype=torch.float32)
    for group, expert in enumerate(experts):
        start, end = offsets[group], offsets[group + 1]
        gate[start:end] = (
            x[start:end].float() @ w_gate[expert].T
            + float(lora_scale) * (gate_low_rank[start:end].float() @ gate_lora_b[expert].float().T)
        )
        up[start:end] = (
            x[start:end].float() @ w_up[expert].T
            + float(lora_scale) * (up_low_rank[start:end].float() @ up_lora_b[expert].float().T)
        )
    sigmoid = torch.sigmoid(gate)
    act = torch.nn.functional.silu(gate) * up
    dgate = dact.float() * up * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
    dup = dact.float() * torch.nn.functional.silu(gate)
    grad_down_a = None
    if dS_down is not None:
        if down_dropout_p > 0.0:
            assert down_mask_bool is not None
            act_for_down = torch.where(
                down_mask_bool,
                act * (1.0 / (1.0 - float(down_dropout_p))),
                torch.zeros_like(act),
            )
        else:
            act_for_down = act
        grad_down_a = torch.zeros(
            (gate_up_weight_cpu.shape[0], dS_down.shape[1], i),
            device=device,
            dtype=torch.float32,
        )

    # Match the native BF16 contract: grad_pair_window stores BF16 dgate/dup and
    # dX is returned as BF16.
    dgate_bf16 = dgate.to(torch.bfloat16)
    dup_bf16 = dup.to(torch.bfloat16)
    dx = torch.empty((m, h), device=device, dtype=torch.float32)
    for group, expert in enumerate(experts):
        start, end = offsets[group], offsets[group + 1]
        dx[start:end] = dgate_bf16[start:end].float() @ w_gate[expert] + dup_bf16[start:end].float() @ w_up[expert]
        if grad_down_a is not None:
            grad_down_a[expert] += dS_down[start:end].float().T @ act_for_down[start:end]
    return dx.to(torch.bfloat16), dgate_bf16, dup_bf16, None if grad_down_a is None else grad_down_a.to(torch.bfloat16)


def _max_err(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    got_f = got.float()
    ref_f = ref.float()
    abs_err = (got_f - ref_f).abs()
    rel = abs_err / ref_f.abs().clamp_min(1e-6)
    return {"max_abs": float(abs_err.max().item()) if abs_err.numel() else 0.0, "max_rel": float(rel.max().item()) if rel.numel() else 0.0}


def _passes_close(got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> bool:
    if got.numel() == 0:
        return True
    return bool(torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol))


class _FakeQwen3Experts(nn.Module):
    def __init__(self, *, num_experts: int, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, device="cuda", dtype=torch.bfloat16) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim, device="cuda", dtype=torch.bfloat16) * 0.02)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_FakeQwen3Experts is only a source container for AsymQwen3Experts")


def _copy_lora_params(lhs: nn.Module, rhs: nn.Module, *, seed: int) -> None:
    with torch.no_grad():
        for name, param in lhs.named_parameters():
            if "lora_" not in name:
                continue
            other = dict(rhs.named_parameters())[name]
            generator = torch.Generator(device=param.device)
            generator.manual_seed(seed + len(name))
            value = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator) * 0.01
            param.copy_(value)
            other.copy_(value)


def _build_top1_routing(counts: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices: list[int] = []
    for expert, count in enumerate(counts):
        indices.extend([expert] * int(count))
    if not indices:
        indices = [0]
    top_k_index = torch.tensor(indices, device=device, dtype=torch.long).view(-1, 1)
    top_k_weights = torch.ones((len(indices), 1), device=device, dtype=torch.bfloat16)
    return top_k_index, top_k_weights


def run_python_integration_case(args: argparse.Namespace) -> dict[str, Any]:
    from asym_gemm.training.qwen3_moe import AsymQwen3Experts

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = _case_config(args.case)
    torch.manual_seed(args.seed)
    e, h, i, r = int(cfg["E"]), int(cfg["H"]), int(cfg["I"]), int(cfg["R"])
    counts = [int(v) for v in cfg["counts"]]  # type: ignore[index]
    source_ref = _FakeQwen3Experts(num_experts=e, hidden_dim=h, intermediate_dim=i)
    source_current = _FakeQwen3Experts(num_experts=e, hidden_dim=h, intermediate_dim=i)
    source_native = _FakeQwen3Experts(num_experts=e, hidden_dim=h, intermediate_dim=i)
    source_current.load_state_dict(source_ref.state_dict())
    source_native.load_state_dict(source_ref.state_dict())
    reference = AsymQwen3Experts(
        source_ref,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=r,
        lora_alpha=2.0 * r,
        lora_dropout=float(args.lora_dropout),
        expert_recompute_policy="none",
        init_lora_weights="peft",
    ).to(device)
    current = AsymQwen3Experts(
        source_current,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=r,
        lora_alpha=2.0 * r,
        lora_dropout=float(args.lora_dropout),
        expert_recompute_policy=args.expert_policy,
        init_lora_weights="peft",
    ).to(device)
    candidate = AsymQwen3Experts(
        source_native,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=r,
        lora_alpha=2.0 * r,
        lora_dropout=float(args.lora_dropout),
        expert_recompute_policy=args.expert_policy,
        init_lora_weights="peft",
    ).to(device)
    _copy_lora_params(reference, candidate, seed=args.seed + 97)
    with torch.no_grad():
        ref_params = dict(reference.named_parameters())
        for name, param in current.named_parameters():
            if "lora_" in name:
                param.copy_(ref_params[name])
    reference.train()
    current.train()
    candidate.train()
    top_k_index, top_k_weights = _build_top1_routing(counts, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    x_ref = torch.randn((top_k_index.shape[0], h), device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    x_current = x_ref.detach().clone().requires_grad_(True)
    x_native = x_ref.detach().clone().requires_grad_(True)

    old_env = os.environ.get("ASYM_QWEN3_GATE_UP_WINDOWED_BWD")
    current_ms = 0.0
    native_ms = 0.0
    try:
        os.environ["ASYM_QWEN3_GATE_UP_WINDOWED_BWD"] = "0"
        torch.manual_seed(args.seed + 313)
        out_ref = reference(x_ref, top_k_index, top_k_weights)
        grad_out = torch.randn_like(out_ref)

        os.environ["ASYM_QWEN3_GATE_UP_WINDOWED_BWD"] = "0"
        torch.manual_seed(args.seed + 313)
        current_start = time.perf_counter()
        out_current = current(x_current, top_k_index, top_k_weights)
        out_current.backward(grad_out)
        torch.cuda.synchronize(device)
        current_ms = (time.perf_counter() - current_start) * 1000.0

        os.environ["ASYM_QWEN3_GATE_UP_WINDOWED_BWD"] = "1"
        torch.manual_seed(args.seed + 313)
        native_start = time.perf_counter()
        out_native = candidate(x_native, top_k_index, top_k_weights)
        out_native.backward(grad_out)
        torch.cuda.synchronize(device)
        native_ms = (time.perf_counter() - native_start) * 1000.0
        out_ref.backward(grad_out)
    finally:
        if old_env is None:
            os.environ.pop("ASYM_QWEN3_GATE_UP_WINDOWED_BWD", None)
        else:
            os.environ["ASYM_QWEN3_GATE_UP_WINDOWED_BWD"] = old_env

    errors: dict[str, dict[str, float]] = {
        "output": _max_err(out_native, out_ref),
        "current_output": _max_err(out_current, out_ref),
        "input_grad": _max_err(x_native.grad, x_ref.grad),
        "current_input_grad": _max_err(x_current.grad, x_ref.grad),
    }
    close_checks = {
        "output": _passes_close(out_native, out_ref, atol=args.max_abs_tol, rtol=args.max_rel_tol),
        "current_output": _passes_close(out_current, out_ref, atol=args.max_abs_tol, rtol=args.max_rel_tol),
        "input_grad": _passes_close(x_native.grad, x_ref.grad, atol=args.max_abs_tol, rtol=args.max_rel_tol),
        "current_input_grad": _passes_close(x_current.grad, x_ref.grad, atol=args.max_abs_tol, rtol=args.max_rel_tol),
    }
    native_params = dict(candidate.named_parameters())
    for name, param in reference.named_parameters():
        if "lora_" not in name:
            continue
        key = f"grad:{name}"
        errors[key] = _max_err(native_params[name].grad, param.grad)
        close_checks[key] = _passes_close(native_params[name].grad, param.grad, atol=args.max_abs_tol, rtol=args.max_rel_tol)
    stats = dict(getattr(candidate, "_last_gate_up_windowed_bwd_stats", {}))
    result = {
        "stage": args.stage,
        "op": args.op,
        "case": args.case,
        "device": str(device),
        "expert_policy": args.expert_policy,
        "lora_dropout": float(args.lora_dropout),
        "shape": {"E": e, "H": h, "I": i, "R": r, "M_selected": int(stats.get("selected_recompute_rows", 0))},
        "latency": {
            "median_latency_ms": native_ms,
            "p50_step_ms": native_ms,
            "p95_step_ms": native_ms,
            "current_selected_region_ms": current_ms,
            "native_selected_region_ms": native_ms,
            "native_kernel_total_ms": float(stats.get("native_total_ms", 0.0)),
            "selected_region_speedup": (current_ms / native_ms) if native_ms > 0.0 else 0.0,
        },
        "errors": errors,
        "close_checks": close_checks,
        "max_abs_error": max(v["max_abs"] for v in errors.values()),
        "max_rel_error": max(v["max_rel"] for v in errors.values()),
        "passed": bool(all(close_checks.values()) and stats),
        "stats": {
            **stats,
            "old_selected_base_dx_rows": int(stats.get("old_selected_base_dx_rows", 0)),
            "new_selected_base_dx_rows": int(stats.get("new_selected_base_dx_rows", 0)),
            "native_kernel_consumed_down_dropout_masks": bool(float(args.lora_dropout) > 0.0 and int(stats.get("down_mask_reads_bytes", 0)) > 0),
            "native_call_count": 1 if stats else 0,
        },
        "ncu": {},
        "nsys": {},
    }
    return result


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    import asym_gemm

    if not hasattr(asym_gemm, "qwen3_gate_up_recompute_bwd_sm100_bf16_windowed"):
        raise RuntimeError("native qwen3_gate_up_recompute_bwd_sm100_bf16_windowed API is missing; rebuild asym_gemm._C")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = _case_config(args.case)
    if args.p is not None:
        cfg["P"] = int(args.p)
    if args.q is not None:
        cfg["Q"] = int(args.q)
    if args.bm is not None:
        cfg["BM"] = int(args.bm)
    if args.bk is not None:
        cfg["BK"] = int(args.bk)
    if args.g_work is not None:
        cfg["G"] = int(args.g_work)

    torch.manual_seed(args.seed)
    e, h, i, r = int(cfg["E"]), int(cfg["H"]), int(cfg["I"]), int(cfg["R"])
    counts = [int(v) for v in cfg["counts"]]  # type: ignore[index]
    offsets_t, experts_t, offsets, active_experts = _build_offsets_experts(counts, device)
    m = int(offsets[-1])
    x = torch.randn((m, h), device=device, dtype=torch.bfloat16)
    dact = torch.randn((m, i), device=device, dtype=torch.bfloat16)
    gate_low_rank = torch.randn((m, r), device=device, dtype=torch.bfloat16)
    up_low_rank = torch.randn((m, r), device=device, dtype=torch.bfloat16)
    dS_down = torch.randn((m, args.r_down), device=device, dtype=torch.bfloat16) if args.with_down_lora_a else None
    down_mask_bool = None
    down_mask_packed = torch.empty((0, 0), device=device, dtype=torch.uint8)
    if args.with_down_lora_a and args.down_dropout_p > 0.0:
        down_mask_bool = torch.rand((m, i), device=device) >= float(args.down_dropout_p)
        down_mask_packed = asym_gemm.pack_bool_mask_2d(down_mask_bool)
    gate_lora_b = torch.randn((e, i, r), device=device, dtype=torch.bfloat16)
    up_lora_b = torch.randn((e, i, r), device=device, dtype=torch.bfloat16)
    gate_up_weight_cpu = torch.randn((e, 2 * i, h), dtype=torch.bfloat16).pin_memory()

    ref_dx, ref_gate, ref_up, ref_down_A = _reference(
        x,
        dact,
        dS_down,
        down_mask_bool,
        args.down_dropout_p,
        gate_low_rank,
        up_low_rank,
        gate_lora_b,
        up_lora_b,
        gate_up_weight_cpu,
        offsets,
        active_experts,
        args.lora_scale,
    )

    for _ in range(args.warmup_iters):
        asym_gemm.qwen3_gate_up_recompute_bwd_sm100_bf16_windowed(
            x,
            dact,
            gate_low_rank,
            up_low_rank,
            gate_lora_b,
            up_lora_b,
            gate_up_weight_cpu,
            offsets_t,
            experts_t,
            p=int(cfg["P"]),
            q=int(cfg["Q"]),
            bm=int(cfg["BM"]),
            bk=int(cfg["BK"]),
            g_work=int(cfg["G"]),
            lora_scale=float(args.lora_scale),
            mode=args.mode,
            return_stats=True,
            dS_down_sel=dS_down if dS_down is not None else torch.Tensor(),
            down_mask_packed=down_mask_packed,
            down_dropout_p=float(args.down_dropout_p) if dS_down is not None else 0.0,
        )
    torch.cuda.synchronize(device)

    latencies: list[float] = []
    last_stats: dict[str, Any] = {}
    got_dx = got_gate = got_up = None
    got_down_A = None
    for _ in range(args.latency_iters):
        start = time.perf_counter()
        native_result = asym_gemm.qwen3_gate_up_recompute_bwd_sm100_bf16_windowed(
            x,
            dact,
            gate_low_rank,
            up_low_rank,
            gate_lora_b,
            up_lora_b,
            gate_up_weight_cpu,
            offsets_t,
            experts_t,
            p=int(cfg["P"]),
            q=int(cfg["Q"]),
            bm=int(cfg["BM"]),
            bk=int(cfg["BK"]),
            g_work=int(cfg["G"]),
            lora_scale=float(args.lora_scale),
            mode=args.mode,
            return_stats=True,
            dS_down_sel=dS_down if dS_down is not None else torch.Tensor(),
            down_mask_packed=down_mask_packed,
            down_dropout_p=float(args.down_dropout_p) if dS_down is not None else 0.0,
        )
        if dS_down is not None:
            got_dx, got_gate, got_up, got_down_A, last_stats = native_result
        else:
            got_dx, got_gate, got_up, last_stats = native_result
            got_down_A = None
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)

    assert got_dx is not None and got_gate is not None and got_up is not None
    errors = {
        "grad_x_base_sel": _max_err(got_dx, ref_dx),
        "grad_gate_sel": _max_err(got_gate, ref_gate),
        "grad_up_sel": _max_err(got_up, ref_up),
    }
    if got_down_A is not None and ref_down_A is not None:
        errors["grad_down_lora_A_sel"] = _max_err(got_down_A, ref_down_A)
    max_abs_error = max(v["max_abs"] for v in errors.values())
    max_rel_error = max(v["max_rel"] for v in errors.values())
    close_checks = {
        "grad_x_base_sel": _passes_close(got_dx, ref_dx, atol=args.max_abs_tol, rtol=args.max_rel_tol),
        "grad_gate_sel": _passes_close(got_gate, ref_gate, atol=args.max_abs_tol, rtol=args.max_rel_tol),
        "grad_up_sel": _passes_close(got_up, ref_up, atol=args.max_abs_tol, rtol=args.max_rel_tol),
    }
    if got_down_A is not None and ref_down_A is not None:
        close_checks["grad_down_lora_A_sel"] = _passes_close(
            got_down_A,
            ref_down_A,
            atol=args.max_abs_tol,
            rtol=args.max_rel_tol,
        )
    passed = all(close_checks.values())
    last_stats = dict(last_stats)
    result = {
        "stage": args.stage,
        "op": args.op,
        "case": args.case,
        "device": str(device),
        "config": {"mode": args.mode, "p": int(cfg["P"]), "q": int(cfg["Q"]), "bm": int(cfg["BM"]), "bk": int(cfg["BK"]), "g_work": int(cfg["G"])},
        "down_lora_A": {"enabled": bool(args.with_down_lora_a), "r_down": int(args.r_down), "dropout_p": float(args.down_dropout_p)},
        "shape": {"E": e, "H": h, "I": i, "R": r, "M_selected": m, "active_experts": len(active_experts)},
        "latency": {
            "median_latency_ms": float(torch.tensor(latencies).median().item()),
            "p50_step_ms": float(torch.tensor(latencies).median().item()),
            "p95_step_ms": float(torch.tensor(latencies).quantile(0.95).item()),
        },
        "errors": errors,
        "close_checks": close_checks,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "passed": bool(passed),
        "stats": last_stats,
        "ncu": {},
        "nsys": {},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen3 gate/up windowed backward kernels.")

    run = parser.add_argument_group("run selection")
    run.add_argument("--stage", default="stage4_native_direct_e2e")
    run.add_argument("--op", default="native_e2e")
    run.add_argument("--case", default="tiny")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--profile-mode", choices=["none", "cuda-events", "nsys", "ncu"], default="none")
    run.add_argument("--output-dir", default="profiling/qwen3_gate_up_windowed_bwd/manual")

    kernel = parser.add_argument_group("kernel shape")
    kernel.add_argument("--mode", default="cache_first_window")
    kernel.add_argument("--p", type=int)
    kernel.add_argument("--q", type=int)
    kernel.add_argument("--bm", type=int)
    kernel.add_argument("--bk", type=int)
    kernel.add_argument("--g-work", type=int)

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--lora-scale", type=float, default=0.5)
    lora.add_argument("--lora-dropout", type=float, default=0.0)
    lora.add_argument("--expert-policy", default="tok-le1024")
    lora.add_argument("--with-down-lora-a", action="store_true", help="Validate selected dA_down output from the native op")
    lora.add_argument("--r-down", type=int, default=8)
    lora.add_argument("--down-dropout-p", type=float, default=0.0)

    validation = parser.add_argument_group("validation")
    validation.add_argument("--seed", type=int, default=0)
    validation.add_argument("--warmup-iters", type=int, default=1)
    validation.add_argument("--latency-iters", type=int, default=1)
    validation.add_argument("--max-abs-tol", type=float, default=5.0)
    validation.add_argument("--max-rel-tol", type=float, default=2.0e-2)
    args = parser.parse_args()
    result = run_python_integration_case(args) if args.op == "python_integration" else run_case(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "validation.md").write_text(
        "\n".join(
            [
                f"# {args.stage} {args.op}",
                "",
                f"case: `{args.case}`",
                f"passed: `{result['passed']}`",
                f"max_abs_error: `{result['max_abs_error']}`",
                f"max_rel_error: `{result['max_rel_error']}`",
                f"median_latency_ms: `{result['latency']['median_latency_ms']}`",
                "",
            ]
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
