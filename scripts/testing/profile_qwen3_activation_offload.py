#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as F

import asym_gemm
from asym_gemm.training.activation_offload import ActivationOffloadManager
import asym_gemm.training.qwen3_moe as qwen3_moe
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
    range_timing: dict
    transfer_timing: dict


class _TimingRecorder:
    def __init__(self, *, sync_cuda: bool) -> None:
        self.sync_cuda = bool(sync_cuda)
        self.enabled = False
        self._rows: dict[str, dict[str, float | int]] = {}
        self._stack: list[list[Any]] = []

    def clear(self) -> None:
        self._rows.clear()
        self._stack.clear()

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def record(self, name: str, *, enabled: bool | None = None) -> Iterator[None]:
        active = self.enabled if enabled is None else bool(enabled and self.enabled)
        if not active:
            yield
            return
        self._sync()
        start = time.perf_counter()
        self._stack.append([str(name), start, 0.0])
        try:
            yield
        finally:
            self._sync()
            end = time.perf_counter()
            current_name, current_start, child_ms = self._stack.pop()
            elapsed_ms = (end - float(current_start)) * 1000.0
            self_ms = max(0.0, elapsed_ms - float(child_ms))
            row = self._rows.setdefault(
                str(current_name),
                {"count": 0, "total_ms": 0.0, "self_ms": 0.0, "max_ms": 0.0},
            )
            row["count"] = int(row["count"]) + 1
            row["total_ms"] = float(row["total_ms"]) + elapsed_ms
            row["self_ms"] = float(row["self_ms"]) + self_ms
            row["max_ms"] = max(float(row["max_ms"]), elapsed_ms)
            if self._stack:
                self._stack[-1][2] = float(self._stack[-1][2]) + elapsed_ms

    def summary(self, *, iters: int) -> dict[str, Any]:
        rows = []
        denom = max(1, int(iters))
        for name, row in self._rows.items():
            total_ms = float(row["total_ms"])
            self_ms = float(row["self_ms"])
            count = int(row["count"])
            rows.append(
                {
                    "name": name,
                    "count": count,
                    "total_ms": total_ms,
                    "self_ms": self_ms,
                    "avg_ms": total_ms / max(1, count),
                    "avg_self_ms": self_ms / max(1, count),
                    "per_step_ms": total_ms / denom,
                    "per_step_self_ms": self_ms / denom,
                    "max_ms": float(row["max_ms"]),
                }
            )
        rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)
        return {
            "enabled": bool(self._rows),
            "sync_cuda": self.sync_cuda,
            "rows": rows,
            "total_ms": sum(float(row["total_ms"]) for row in self._rows.values()),
        }


@contextmanager
def _patched_breakdown_timers(
    range_recorder: _TimingRecorder | None,
    transfer_recorder: _TimingRecorder | None,
) -> Iterator[None]:
    original_prof_range = qwen3_moe.prof_range
    original_offload = ActivationOffloadManager.offload
    original_stage = ActivationOffloadManager.stage
    original_stage_concat = ActivationOffloadManager.stage_concat_columns

    def timed_prof_range(name: str, *, enabled: bool | None = None):
        if range_recorder is None:
            return original_prof_range(name, enabled=enabled)
        return range_recorder.record(name, enabled=enabled)

    def timed_offload(self: ActivationOffloadManager, tensor: torch.Tensor, tag: str):
        if transfer_recorder is None:
            return original_offload(self, tensor, tag)
        with transfer_recorder.record(f"transfer.offload.{tag}"):
            return original_offload(self, tensor, tag)

    def timed_stage(self: ActivationOffloadManager, handle, *, tag: str | None = None):
        if transfer_recorder is None:
            return original_stage(self, handle, tag=tag)
        stage_tag = handle.tag if tag is None else tag
        with transfer_recorder.record(f"transfer.stage.{stage_tag}"):
            return original_stage(self, handle, tag=tag)

    def timed_stage_concat(self: ActivationOffloadManager, left, right, *, tag: str):
        if transfer_recorder is None:
            return original_stage_concat(self, left, right, tag=tag)
        with transfer_recorder.record(f"transfer.stage_concat.{tag}"):
            return original_stage_concat(self, left, right, tag=tag)

    if range_recorder is not None:
        qwen3_moe.prof_range = timed_prof_range
    if transfer_recorder is not None:
        ActivationOffloadManager.offload = timed_offload
        ActivationOffloadManager.stage = timed_stage
        ActivationOffloadManager.stage_concat_columns = timed_stage_concat
    try:
        yield
    finally:
        qwen3_moe.prof_range = original_prof_range
        ActivationOffloadManager.offload = original_offload
        ActivationOffloadManager.stage = original_stage
        ActivationOffloadManager.stage_concat_columns = original_stage_concat


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


def _reset_execution_stats(stats: Any) -> None:
    dataclass_fields = getattr(stats, "__dataclass_fields__", None)
    if not isinstance(dataclass_fields, dict):
        return
    for field in fields(stats):
        value = getattr(stats, field.name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            setattr(stats, field.name, 0)
        elif isinstance(value, dict):
            value.clear()


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
        _reset_execution_stats(getattr(model, "stats", None))

        range_recorder = _TimingRecorder(sync_cuda=args.breakdown_sync_cuda) if args.profile_breakdown else None
        transfer_recorder = _TimingRecorder(sync_cuda=args.breakdown_sync_cuda) if args.profile_breakdown else None
        peaks_allocated: list[int] = []
        peaks_reserved: list[int] = []
        losses: list[float] = []
        with _patched_breakdown_timers(range_recorder, transfer_recorder):
            if range_recorder is not None:
                range_recorder.enabled = True
            if transfer_recorder is not None:
                transfer_recorder.enabled = True
            if args.use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats()
                start.record()
                for idx in range(args.iters):
                    losses.append(_step(model, args.seed + 2000 + idx, top_k_index, top_k_weights, args))
                end.record()
                torch.cuda.synchronize()
                peaks_allocated.append(int(torch.cuda.max_memory_allocated()))
                peaks_reserved.append(int(torch.cuda.max_memory_reserved()))
                step_ms = float(start.elapsed_time(end)) / max(1, args.iters)
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
            range_timing=range_recorder.summary(iters=args.iters) if range_recorder is not None else {"enabled": False, "rows": []},
            transfer_timing=transfer_recorder.summary(iters=args.iters) if transfer_recorder is not None else {"enabled": False, "rows": []},
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
    parser.add_argument("--profile-breakdown", action="store_true")
    parser.add_argument("--breakdown-sync-cuda", action=argparse.BooleanOptionalAction, default=True)
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
