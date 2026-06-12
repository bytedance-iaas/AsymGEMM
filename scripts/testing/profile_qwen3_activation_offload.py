#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

import asym_gemm
from asym_gemm.training.qwen3_moe import AsymQwen3Experts


class FakeQwen3Experts(nn.Module):
    def __init__(self, *, num_experts: int, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16) * 0.02
        )
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16) * 0.02)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("profile script uses AsymQwen3Experts directly")


@dataclass
class VariantResult:
    variant: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    step_ms: float
    loss: float
    stats: dict
    activation_offload_stats: dict


def _require_sm100_bf16() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability(0)[0] < 10:
        raise RuntimeError(f"SM100 is required, got capability={torch.cuda.get_device_capability(0)}")
    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        raise RuntimeError("asym_gemm._C does not export m_grouped_bf16_asym_gemm_nt_contiguous")


def _make_routing(tokens: int, top_k: int, num_experts: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    top_k_index = torch.randint(0, num_experts, (tokens, top_k), generator=gen, dtype=torch.long)
    weights = torch.rand(tokens, top_k, generator=gen, dtype=torch.float32)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(dtype=torch.bfloat16)
    return top_k_index.cuda(), weights.cuda()


def _copy_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def _make_lora_state(model: nn.Module, *, seed: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" not in name:
                continue
            gen = torch.Generator(device=param.device)
            gen.manual_seed(seed + len(name))
            state[name] = (torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=gen) * 0.01).detach().clone()
    return state


def _make_model(args: argparse.Namespace, source_state: dict[str, torch.Tensor], lora_state: dict[str, torch.Tensor] | None) -> AsymQwen3Experts:
    source = FakeQwen3Experts(num_experts=args.num_experts, hidden_dim=args.hidden_dim, intermediate_dim=args.intermediate_dim)
    source.load_state_dict(source_state)
    model = AsymQwen3Experts(
        source,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        init_lora_weights="peft",
    )
    model.cuda()
    model.train()
    if lora_state is not None:
        _copy_lora_state(model, lora_state)
    return model


def _step(model: AsymQwen3Experts, x_seed: int, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, args: argparse.Namespace) -> float:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(x_seed)
    x = torch.randn(args.tokens, args.hidden_dim, device="cuda", dtype=torch.bfloat16, generator=gen, requires_grad=True)
    for param in model.parameters():
        param.grad = None
    out = model(x, top_k_index, top_k_weights)
    loss = out.float().square().mean()
    loss.backward()
    return float(loss.detach().cpu().item())


def _profile_variant(
    variant: str,
    model: AsymQwen3Experts,
    *,
    activation_offload: bool,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    args: argparse.Namespace,
) -> VariantResult:
    previous = os.environ.get("ASYMM_EXPERT_ACT_OFFLOAD")
    if activation_offload:
        os.environ["ASYMM_EXPERT_ACT_OFFLOAD"] = "1"
    else:
        os.environ.pop("ASYMM_EXPERT_ACT_OFFLOAD", None)
    try:
        for idx in range(args.warmup):
            _step(model, args.seed + 1000 + idx, top_k_index, top_k_weights, args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        peaks_allocated: list[int] = []
        peaks_reserved: list[int] = []
        losses: list[float] = []
        if args.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            start.record()
            for idx in range(args.iters):
                losses.append(_step(model, args.seed + 2000 + idx, top_k_index, top_k_weights, args))
            end.record()
            torch.cuda.synchronize()
            step_ms = float(start.elapsed_time(end)) / max(1, args.iters)
            peaks_allocated.append(int(torch.cuda.max_memory_allocated()))
            peaks_reserved.append(int(torch.cuda.max_memory_reserved()))
        else:
            elapsed = 0.0
            for idx in range(args.iters):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                losses.append(_step(model, args.seed + 2000 + idx, top_k_index, top_k_weights, args))
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - t0
                peaks_allocated.append(int(torch.cuda.max_memory_allocated()))
                peaks_reserved.append(int(torch.cuda.max_memory_reserved()))
            step_ms = elapsed * 1000.0 / max(1, args.iters)
        return VariantResult(
            variant=variant,
            peak_allocated_bytes=max(peaks_allocated) if peaks_allocated else 0,
            peak_reserved_bytes=max(peaks_reserved) if peaks_reserved else 0,
            step_ms=step_ms,
            loss=losses[-1] if losses else 0.0,
            stats=model.stats.as_dict(),
            activation_offload_stats=dict(getattr(model, "_last_activation_offload_stats", {})),
        )
    finally:
        if previous is None:
            os.environ.pop("ASYMM_EXPERT_ACT_OFFLOAD", None)
        else:
            os.environ["ASYMM_EXPERT_ACT_OFFLOAD"] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Qwen3 expert activation offload against the current AsymGEMM path.")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--intermediate-dim", type=int, default=11008)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use-cuda-events", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    _require_sm100_bf16()
    if args.hidden_dim % 64 != 0 or args.intermediate_dim % 64 != 0:
        raise RuntimeError("hidden_dim and intermediate_dim must be multiples of 64 for activation offload v0")

    torch.manual_seed(args.seed)
    source = FakeQwen3Experts(num_experts=args.num_experts, hidden_dim=args.hidden_dim, intermediate_dim=args.intermediate_dim)
    source_state = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
    top_k_index, top_k_weights = _make_routing(args.tokens, args.top_k, args.num_experts, seed=args.seed + 17)

    baseline = _make_model(args, source_state, None)
    lora_state = _make_lora_state(baseline, seed=args.seed + 31)
    _copy_lora_state(baseline, lora_state)
    current = _profile_variant("current_asym", baseline, activation_offload=False, top_k_index=top_k_index, top_k_weights=top_k_weights, args=args)
    del baseline
    torch.cuda.empty_cache()

    candidate = _make_model(args, source_state, lora_state)
    act = _profile_variant(
        "activation_offload",
        candidate,
        activation_offload=True,
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        args=args,
    )
    del candidate
    torch.cuda.empty_cache()

    peak_delta = act.peak_allocated_bytes - current.peak_allocated_bytes
    reserved_delta = act.peak_reserved_bytes - current.peak_reserved_bytes
    result = {
        "config": vars(args),
        "variants": [asdict(current), asdict(act)],
        "comparison": {
            "peak_allocated_delta_bytes": peak_delta,
            "peak_allocated_delta_pct": (peak_delta / current.peak_allocated_bytes) if current.peak_allocated_bytes else None,
            "peak_reserved_delta_bytes": reserved_delta,
            "peak_reserved_delta_pct": (reserved_delta / current.peak_reserved_bytes) if current.peak_reserved_bytes else None,
            "step_ms_delta": act.step_ms - current.step_ms,
            "slowdown_vs_current": (act.step_ms / current.step_ms) if current.step_ms else None,
            "loss_abs_delta": abs(act.loss - current.loss),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
