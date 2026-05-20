#!/usr/bin/env python3
"""Exclusive-ish M4.1-M4.3 toy profiling reports.

This profiler intentionally reports additive tables.  Rows that cannot be
split safely from PyTorch autograd are kept in explicit *_other buckets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
if __name__ == "__main__":
    # Keep the script directory out of import search while this runs as a file;
    # profiling dependencies may import modules with common names.
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asym_gemm.training.profile_ranges import profile_enabled


QWEN3_14B_CONFIG = {
    "hf_model_id": "Qwen/Qwen3-14B",
    "hf_model_type": "qwen3",
    "hf_num_hidden_layers": 40,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 40,
    "vocab_size": 151936,
}


QWEN3_30B_A3B_CONFIG = {
    "hf_model_id": "Qwen/Qwen3-30B-A3B",
    "hf_model_type": "qwen3_moe",
    "hf_num_hidden_layers": 48,
    "hidden_size": 2048,
    "intermediate_size": 768,
    "moe_intermediate_size": 768,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_shared_experts": 0,
    "vocab_size": 151936,
}


CUSTOM_DENSE_3B_CONFIG = {
    "hf_model_id": "custom/dense-3b",
    "hf_model_type": "dense_3b",
    "hf_num_hidden_layers": 36,
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_attention_heads": 16,
    "vocab_size": 151936,
}


CUSTOM_MOE_3B_CONFIG = {
    "hf_model_id": "custom/moe-3b-active",
    "hf_model_type": "moe_3b_active",
    "hf_num_hidden_layers": 32,
    "hidden_size": 2048,
    "intermediate_size": 1536,
    "moe_intermediate_size": 1536,
    "num_attention_heads": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_shared_experts": 0,
    "vocab_size": 151936,
}


MM_CONFIGS = {
    "mm_1b": {
        "tokens": 64,
        "in_features": 32768,
        "out_features": 32768,
    },
    "mm_3b": {
        "tokens": 64,
        "in_features": 55296,
        "out_features": 55296,
    },
}
MATRIX_1B_CONFIG = MM_CONFIGS["mm_1b"]


MLP_CONFIGS = {
    "mlp_1b": {
        "tokens": 64,
        "in_features": 8192,
        "hidden_features": 65536,
        "out_features": 8192,
    },
    "mlp_3b": {
        "tokens": 64,
        "in_features": 8192,
        "hidden_features": 183040,
        "out_features": 8192,
    },
}
MLP_1B_CONFIG = MLP_CONFIGS["mlp_1b"]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def nbytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.numel() * tensor.element_size())


def requested_tokens(args: argparse.Namespace, default: int) -> int:
    tokens = int(getattr(args, "real_tokens", 0) or 0)
    return tokens if tokens > 0 else int(default)


class StageBook:
    def __init__(self, device: torch.device, *, timing_mode: str = "profile") -> None:
        self.device = device
        self.timing_mode = timing_mode
        self.values: dict[str, float] = defaultdict(float)

    def flush(self) -> None:
        sync(self.device)

    @contextmanager
    def time(self, key: str) -> Iterator[None]:
        should_sync = self.timing_mode == "debug_sync" or key.startswith("step.")
        if should_sync:
            sync(self.device)
        start = time.perf_counter()
        with torch.autograd.profiler.record_function(key):
            if self.device.type == "cuda":
                torch.cuda.nvtx.range_push(key)
            try:
                yield
            finally:
                if should_sync:
                    sync(self.device)
                self.values[key] += time.perf_counter() - start
                if self.device.type == "cuda":
                    torch.cuda.nvtx.range_pop()


class _ProfiledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, prefix: str, book: StageBook) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.prefix = prefix
        ctx.book = book
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None]:
        x, weight = ctx.saved_tensors
        prefix = ctx.prefix
        book = ctx.book
        grad_x = grad_weight = grad_bias = None
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        if ctx.needs_input_grad[0]:
            with book.time(f"backward.{prefix}.input_grad"):
                grad_x = grad_output_2d.matmul(weight).reshape_as(x)
        if ctx.needs_input_grad[1]:
            with book.time(f"backward.{prefix}.weight_grad"):
                grad_weight = grad_output_2d.t().matmul(x_2d)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            with book.time(f"backward.{prefix}.bias_grad"):
                grad_bias = grad_output_2d.sum(dim=0)
        return grad_x, grad_weight, grad_bias, None, None


def profiled_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledLinear.apply(x, weight, bias, prefix, book)


class _ProfiledRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.prefix = prefix
        ctx.book = book
        return F.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            grad_x = grad_output * (x > 0)
        return grad_x, None, None


def profiled_relu(x: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledRelu.apply(x, prefix, book)


class _ProfiledSiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
        gate_f = gate.float()
        up_f = up.float()
        sigmoid = torch.sigmoid(gate_f)
        silu = gate_f * sigmoid
        ctx.save_for_backward(gate_f, up_f, sigmoid, silu)
        ctx.prefix = prefix
        ctx.book = book
        return silu * up_f

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        gate, up, sigmoid, silu = ctx.saved_tensors
        grad = grad_output.float()
        grad_gate = grad_up = None
        if ctx.needs_input_grad[0]:
            with ctx.book.time(f"backward.{ctx.prefix}.gate_activation_grad"):
                grad_gate = grad * up * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
        if ctx.needs_input_grad[1]:
            with ctx.book.time(f"backward.{ctx.prefix}.up_mul_grad"):
                grad_up = grad * silu
        return grad_gate, grad_up, None, None


def profiled_silu_mul(gate: torch.Tensor, up: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledSiluMul.apply(gate, up, prefix, book)


class _ProfiledSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int, prefix: str, book: StageBook) -> torch.Tensor:
        out = torch.softmax(x, dim=dim)
        ctx.save_for_backward(out)
        ctx.dim = dim
        ctx.prefix = prefix
        ctx.book = book
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (out,) = ctx.saved_tensors
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            grad_x = out * (grad_output - (grad_output * out).sum(dim=ctx.dim, keepdim=True))
        return grad_x, None, None, None


def profiled_softmax(x: torch.Tensor, dim: int, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledSoftmax.apply(x, dim, prefix, book)


class _ProfiledMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        ctx.prefix = prefix
        ctx.book = book
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            with ctx.book.time(f"backward.{ctx.prefix}.lhs_grad"):
                grad_a = torch.matmul(grad_output, b.transpose(-2, -1))
        if ctx.needs_input_grad[1]:
            with ctx.book.time(f"backward.{ctx.prefix}.rhs_grad"):
                grad_b = torch.matmul(a.transpose(-2, -1), grad_output)
        return grad_a, grad_b, None, None


def profiled_matmul(a: torch.Tensor, b: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledMatmul.apply(a, b, prefix, book)


class _ProfiledLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        prefix: str,
        book: StageBook,
    ) -> torch.Tensor:
        x_f = x.float()
        weight_f = weight.float()
        bias_f = None if bias is None else bias.float()
        mean = x_f.mean(dim=-1, keepdim=True)
        centered = x_f - mean
        rstd = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + eps)
        normalized = centered * rstd
        out = normalized * weight_f
        if bias_f is not None:
            out = out + bias_f
        ctx.save_for_backward(normalized, rstd, weight_f)
        ctx.prefix = prefix
        ctx.book = book
        return out.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None, None, None]:
        normalized, rstd, weight = ctx.saved_tensors
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            grad_norm = grad_output.float() * weight
            features = float(normalized.shape[-1])
            grad_x = (grad_norm - grad_norm.mean(dim=-1, keepdim=True) - normalized * (grad_norm * normalized).mean(dim=-1, keepdim=True)) * rstd
        return grad_x.to(dtype=grad_output.dtype), None, None, None, None, None


def profiled_layer_norm(module: torch.nn.LayerNorm, x: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledLayerNorm.apply(x, module.weight, module.bias, float(module.eps), prefix, book)


def profiled_frozen_layer_norm(module: torch.nn.Module, x: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledLayerNorm.apply(x, module.frozen_weight, module.frozen_bias, float(module.eps), prefix, book)


class _ProfiledResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor, scale: float, prefix: str, book: StageBook) -> torch.Tensor:
        ctx.scale = scale
        ctx.prefix = prefix
        ctx.book = book
        return (a.float() + scale * b.float()).to(dtype=a.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        grad_a = grad_b = None
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            if ctx.needs_input_grad[0]:
                grad_a = grad_output
            if ctx.needs_input_grad[1]:
                grad_b = (grad_output.float() * float(ctx.scale)).to(dtype=grad_output.dtype)
        return grad_a, grad_b, None, None, None


def profiled_residual_add(a: torch.Tensor, b: torch.Tensor, prefix: str, book: StageBook, scale: float = 1.0) -> torch.Tensor:
    return _ProfiledResidualAdd.apply(a, b, scale, prefix, book)


class _ProfiledScaleCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float, dtype: torch.dtype, prefix: str, book: StageBook) -> torch.Tensor:
        ctx.scale = scale
        ctx.prefix = prefix
        ctx.book = book
        return (x * scale).to(dtype=dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        with ctx.book.time(f"backward.{ctx.prefix}.scale_cast_grad"):
            grad_x = grad_output.float() * float(ctx.scale)
        return grad_x, None, None, None, None


def profiled_scale_cast(x: torch.Tensor, scale: float, dtype: torch.dtype, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledScaleCast.apply(x, scale, dtype, prefix, book)


class _ProfiledPackTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, metadata: Any, mode: str, book: StageBook) -> torch.Tensor:
        ctx.route_metadata = metadata
        ctx.mode = mode
        ctx.book = book
        flat = hidden.reshape(metadata.num_tokens, -1)
        if mode == "contiguous":
            out = flat.index_select(0, metadata.token_indices).reshape(metadata.num_routes, *hidden.shape[1:]).contiguous()
        else:
            packed = flat.index_select(0, metadata.token_indices.reshape(-1))
            packed = packed.reshape(metadata.num_experts, metadata.max_routes_per_expert, *hidden.shape[1:])
            mask = metadata.valid_mask.reshape(metadata.num_experts, metadata.max_routes_per_expert, *([1] * (hidden.dim() - 1)))
            out = packed * mask.to(dtype=packed.dtype)
        ctx.hidden_shape = tuple(hidden.shape)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        metadata = ctx.route_metadata
        with ctx.book.time("backward.pack_tokens.grad"):
            grad_flat = torch.zeros(
                (metadata.num_tokens, int(torch.tensor(ctx.hidden_shape[1:]).prod().item())),
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            if ctx.mode == "contiguous":
                grad_flat.index_add_(0, metadata.token_indices, grad_output.reshape(metadata.num_routes, -1))
            else:
                grad = grad_output.reshape(metadata.num_experts * metadata.max_routes_per_expert, -1)
                mask = metadata.valid_mask.reshape(-1)
                grad_flat.index_add_(0, metadata.token_indices.reshape(-1)[mask], grad[mask])
            grad_hidden = grad_flat.reshape(ctx.hidden_shape)
        return grad_hidden, None, None, None


def profiled_pack_tokens(hidden: torch.Tensor, metadata: Any, mode: str, book: StageBook) -> torch.Tensor:
    return _ProfiledPackTokens.apply(hidden, metadata, mode, book)


class _ProfiledScatterTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, expert_output: torch.Tensor, metadata: Any, mode: str, book: StageBook) -> torch.Tensor:
        ctx.route_metadata = metadata
        ctx.mode = mode
        ctx.book = book
        ctx.expert_shape = tuple(expert_output.shape)
        if mode == "contiguous":
            flat = expert_output.reshape(metadata.num_routes, -1)
            weights = metadata.routing_weights.reshape(metadata.num_routes, 1)
            weighted = flat * weights
            out = torch.zeros((metadata.num_tokens, flat.shape[1]), device=expert_output.device, dtype=weighted.dtype)
            out.index_add_(0, metadata.token_indices, weighted)
            return out.reshape(metadata.num_tokens, *expert_output.shape[1:])
        flat = expert_output.reshape(metadata.num_experts * metadata.max_routes_per_expert, -1)
        flat_weights = metadata.routing_weights.reshape(-1, 1)
        flat_tokens = metadata.token_indices.reshape(-1)
        flat_mask = metadata.valid_mask.reshape(-1)
        weighted = flat * flat_weights
        out = torch.zeros((metadata.num_tokens, flat.shape[1]), device=expert_output.device, dtype=weighted.dtype)
        out.index_add_(0, flat_tokens[flat_mask], weighted[flat_mask])
        return out.reshape(metadata.num_tokens, *expert_output.shape[2:])

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        metadata = ctx.route_metadata
        with ctx.book.time("backward.scatter_combine.grad"):
            grad_flat = grad_output.reshape(metadata.num_tokens, -1)
            if ctx.mode == "contiguous":
                weights = metadata.routing_weights.reshape(metadata.num_routes, 1)
                grad_expert = grad_flat.index_select(0, metadata.token_indices) * weights
            else:
                flat_tokens = metadata.token_indices.reshape(-1)
                flat_mask = metadata.valid_mask.reshape(-1)
                weights = metadata.routing_weights.reshape(-1, 1)
                grad_expert = torch.zeros(
                    (metadata.num_experts * metadata.max_routes_per_expert, grad_flat.shape[1]),
                    device=grad_output.device,
                    dtype=grad_output.dtype,
                )
                grad_expert[flat_mask] = grad_flat.index_select(0, flat_tokens[flat_mask]) * weights[flat_mask]
            grad_expert = grad_expert.reshape(ctx.expert_shape)
        return grad_expert, None, None, None


def profiled_scatter_tokens(expert_output: torch.Tensor, metadata: Any, mode: str, book: StageBook) -> torch.Tensor:
    return _ProfiledScatterTokens.apply(expert_output, metadata, mode, book)


class _ProfiledMSELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, prediction: torch.Tensor, target: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
        diff = prediction.float() - target.float()
        ctx.save_for_backward(diff)
        ctx.prefix = prefix
        ctx.book = book
        return diff.pow(2).mean()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (diff,) = ctx.saved_tensors
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            grad_prediction = grad_output.float() * (2.0 / float(diff.numel())) * diff
        return grad_prediction, None, None, None


def profiled_mse_loss(prediction: torch.Tensor, target: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledMSELoss.apply(prediction, target, prefix, book)


class _ProfiledCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, labels: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
        logits_f = logits.float()
        labels_l = labels.to(device=logits.device, dtype=torch.long)
        log_probs = F.log_softmax(logits_f, dim=-1)
        ctx.save_for_backward(torch.exp(log_probs), labels_l)
        ctx.prefix = prefix
        ctx.book = book
        return F.nll_loss(log_probs, labels_l, reduction="mean")

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        probs, labels = ctx.saved_tensors
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            grad_logits = probs
            grad_logits = grad_logits.clone()
            grad_logits.scatter_add_(-1, labels.unsqueeze(-1), -torch.ones_like(labels, dtype=grad_logits.dtype).unsqueeze(-1))
            grad_logits.mul_(grad_output.float() / float(labels.numel()))
        return grad_logits, None, None, None


def profiled_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledCrossEntropyLoss.apply(logits, labels, prefix, book)


def average(values: dict[str, float], steps: int) -> dict[str, float]:
    return {key: value / float(max(1, steps)) for key, value in values.items()}


def reset_execution_stats(stats: Any) -> None:
    for name in (
        "asym_forward_calls",
        "asym_dx_calls",
        "staged_forward_calls",
        "staged_dx_calls",
        "torch_forward_calls",
        "torch_dx_calls",
    ):
        if hasattr(stats, name):
            setattr(stats, name, 0)
    fallback_reasons = getattr(stats, "fallback_reasons", None)
    if isinstance(fallback_reasons, dict):
        fallback_reasons.clear()


def raw_seconds_without_individual_calls(values: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in sorted(values.items())
        if not (key.rpartition(".call_")[1] and key.rpartition(".call_")[2].isdigit())
    }


def call_group_seconds_per_step(values: dict[str, float]) -> dict[str, float]:
    groups: dict[str, float] = defaultdict(float)
    for key, value in values.items():
        head, sep, tail = key.rpartition(".call_")
        if sep and tail.isdigit():
            groups[head] += value
    return dict(sorted(groups.items()))


def rows_from_keys(values: dict[str, float], parent: float, keys: list[str], *, residual_name: str = "other_unattributed") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accounted = 0.0
    for key in keys:
        seconds = float(values.get(key, 0.0))
        accounted += seconds
        rows.append(
            {
                "name": key,
                "seconds": seconds,
                "milliseconds": seconds * 1000.0,
                "percent": 0.0 if parent <= 0.0 else seconds * 100.0 / parent,
            }
        )
    other = max(0.0, parent - accounted)
    rows.append(
        {
            "name": residual_name,
            "seconds": other,
            "milliseconds": other * 1000.0,
            "percent": 0.0 if parent <= 0.0 else other * 100.0 / parent,
        }
    )
    return rows


def table(total: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_seconds": total,
        "total_milliseconds": total * 1000.0,
        "rows": rows,
        "sum_seconds": sum(float(row["seconds"]) for row in rows),
        "sum_percent": sum(float(row["percent"]) for row in rows),
    }


def memory_report(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    host_w = 0
    host_w_t = 0
    pinned_w = 0
    pinned_w_t = 0
    pinned = 0
    seen_host_weights: set[int] = set()
    for module in model.modules():
        host_weight = getattr(module, "host_weight", None)
        if host_weight is None:
            base = getattr(module, "base", None) or getattr(module, "base_layer", None)
            host_weight = getattr(base, "host_weight", None)
        if host_weight is None:
            continue
        host_id = id(host_weight)
        if host_id in seen_host_weights:
            continue
        seen_host_weights.add(host_id)
        weight = getattr(host_weight, "weight", None)
        transpose = getattr(host_weight, "_transpose", None)
        if isinstance(weight, torch.Tensor):
            weight_bytes = nbytes(weight)
            host_w += weight_bytes
            if weight.is_pinned():
                pinned_w += weight_bytes
        if isinstance(transpose, torch.Tensor):
            transpose_bytes = nbytes(transpose)
            host_w_t += transpose_bytes
            if transpose.is_pinned():
                pinned_w_t += transpose_bytes
        pinned += int(getattr(host_weight, "pinned_cpu_bytes", 0))
    gpu_params = sum(nbytes(param) for param in model.parameters() if param.device.type == "cuda")
    gpu_buffers = sum(nbytes(buffer) for buffer in model.buffers() if buffer.device.type == "cuda")
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "gpu": {
            "peak_hbm_bytes": peak,
            "parameter_bytes": gpu_params,
            "buffer_bytes": gpu_buffers,
            "unattributed_peak_bytes": max(0, peak - gpu_params - gpu_buffers),
        },
        "cpu": {
            "host_w_bytes": host_w,
            "host_w_t_bytes": host_w_t,
            "pinned_w_bytes": pinned_w,
            "pinned_w_t_bytes": pinned_w_t,
            "pinned_total_bytes": pinned,
            "pinned_w_t_percent_of_host_weight": 0.0 if pinned_w + pinned_w_t <= 0 else pinned_w_t * 100.0 / (pinned_w + pinned_w_t),
            "host_w_t_percent_of_host_weight": 0.0 if host_w + host_w_t <= 0 else host_w_t * 100.0 / (host_w + host_w_t),
        },
    }


@contextmanager
def patch_base_dispatch(book: StageBook) -> Iterator[None]:
    import importlib

    frozen_linear = importlib.import_module("asym_gemm.training.frozen_linear")

    original = frozen_linear._dispatch_nt
    counts: dict[str, int] = defaultdict(int)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        phase = str(kwargs.get("phase", "forward"))
        profile_label = str(kwargs.get("profile_label") or "")
        if phase == "dx":
            base_key = profile_label or "backward.base_dx_asymgemm"
            idx = counts[base_key]
            counts[base_key] += 1
            key = f"{base_key}.call_{idx}"
        else:
            idx = counts["forward"]
            counts["forward"] += 1
            key = f"forward.base_frozen_asymgemm.call_{idx}"
        with book.time(key):
            return original(*args, **kwargs)

    frozen_linear._dispatch_nt = wrapper
    try:
        yield
    finally:
        frozen_linear._dispatch_nt = original


def patch_mlp_forward(book: StageBook) -> list[tuple[Any, str, Any]]:
    import asym_gemm.training.mlp as mlp

    originals: list[tuple[Any, str, Any]] = []
    original = mlp.AsymLoRALinear.forward
    counter: dict[str, int] = defaultdict(int)

    def timed_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        idx = counter["linear"]
        counter["linear"] += 1
        prefix = "fc1" if idx % 2 == 0 else "fc2"
        with book.time(f"forward.{prefix}.base_frozen_asymgemm"):
            self.base.profile_name = prefix
            base = self.base(x)
        with book.time(f"forward.{prefix}.lora_A"):
            low_rank = profiled_linear(x.float(), self.lora_a, None, f"{prefix}.lora_A", book)
        with book.time(f"forward.{prefix}.lora_B"):
            lora_raw = profiled_linear(low_rank, self.lora_b, None, f"{prefix}.lora_B", book)
        with book.time(f"forward.{prefix}.add_cast_scale"):
            lora = profiled_scale_cast(lora_raw, float(self.scaling), base.dtype, f"{prefix}.add_cast_scale", book)
            return profiled_residual_add(base, lora, f"{prefix}.base_lora_add", book)

    originals.append((mlp.AsymLoRALinear, "forward", original))
    mlp.AsymLoRALinear.forward = timed_forward
    mlp_original = mlp.AsymMLP.forward

    def timed_mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        with book.time("forward.activation_relu"):
            hidden = profiled_relu(hidden, "activation_relu", book)
        return self.fc2(hidden)

    originals.append((mlp.AsymMLP, "forward", mlp_original))
    mlp.AsymMLP.forward = timed_mlp_forward
    return originals


def restore(originals: list[tuple[Any, str, Any]]) -> None:
    for obj, attr, original in reversed(originals):
        setattr(obj, attr, original)


def profile_mlp(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.mlp import AsymMLP, lora_parameters

    clear(device)
    stats = AsymExecutionStats()
    default_tokens, in_features, hidden, out_features, rank = (64, 128, 256, 128, 8) if device.type == "cuda" else (4, 16, 32, 16, 4)
    tokens = requested_tokens(args, default_tokens)
    torch.manual_seed(0)
    model = AsymMLP(
        torch.randn(hidden, in_features, dtype=dtype),
        torch.randn(out_features, hidden, dtype=dtype),
        rank=rank,
        alpha=16.0,
        backend=args.backend,
        stats=stats,
        device=device,
        dtype=dtype,
        precision=args.precision,
    )
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(tokens, in_features, device=device, dtype=dtype, requires_grad=True),
            torch.randn(tokens, out_features, device=device, dtype=dtype),
        )

    book = StageBook(device, timing_mode=args.timing_mode)
    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return profiled_mse_loss(prediction.float(), target.float(), "loss.mse", book)

    originals = patch_mlp_forward(book)
    try:
        run_profile_steps(
            book,
            model,
            optimizer,
            make_batch,
            lambda m, x: m(x),
            loss_fn,
            args,
            reset_stats=lambda: reset_execution_stats(stats),
        )
    finally:
        restore(originals)

    avg = average(book.values, args.measure_steps)
    return build_report(
        "m4_1_mlp",
        model,
        device,
        avg,
        stats.as_dict(),
        forward_keys=[
            "forward.fc1.base_frozen_asymgemm",
            "forward.fc1.lora_A",
            "forward.fc1.lora_B",
            "forward.fc1.add_cast_scale",
            "forward.activation_relu",
            "forward.fc2.base_frozen_asymgemm",
            "forward.fc2.lora_A",
            "forward.fc2.lora_B",
            "forward.fc2.add_cast_scale",
        ],
        backward_group_prefixes=[
            "backward.loss.mse",
            "backward.fc2.base_dx_asymgemm",
            "backward.fc2.base_lora_add",
            "backward.fc2.add_cast_scale",
            "backward.fc2.lora_B",
            "backward.fc2.lora_A",
            "backward.activation_relu",
            "backward.fc1.base_dx_asymgemm",
            "backward.fc1.base_lora_add",
            "backward.fc1.add_cast_scale",
            "backward.fc1.lora_B",
            "backward.fc1.lora_A",
        ],
        config={"tokens": tokens, "in_features": in_features, "hidden": hidden, "out_features": out_features, "rank": rank},
    )


def run_fundamental_steps(
    book: StageBook,
    make_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    args: argparse.Namespace,
    reset_stats: Callable[[], None] | None = None,
) -> None:
    with profile_enabled(True):
        for _ in range(args.warmup_steps):
            batch, target = make_batch()
            loss = loss_fn(forward_fn(batch), target)
            loss.backward()
            sync(book.device)
        if reset_stats is not None:
            reset_stats()
        book.values.clear()

        with patch_base_dispatch(book):
            for _ in range(args.measure_steps):
                with book.time("step.input_preparation"):
                    batch, target = make_batch()
                with book.time("step.forward"):
                    prediction = forward_fn(batch)
                with book.time("step.loss"):
                    loss = loss_fn(prediction, target)
                with book.time("step.backward"):
                    loss.backward()
                with book.time("step.optimizer"):
                    pass
                sync(book.device)
                book.flush()


def _fundamental_config(raw: dict[str, int]) -> dict[str, Any]:
    config = dict(raw)
    if "hidden_features" in config:
        params = config["hidden_features"] * (config["in_features"] + config["out_features"])
    else:
        params = config["in_features"] * config["out_features"]
    config["base_parameter_count"] = int(params)
    config["base_parameter_billions"] = float(params) / 1_000_000_000.0
    config["profile_goal"] = "fundamental_parameter_asymgemm"
    return config


def profile_mm(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, workload: str) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymFrozenLinear

    clear(device)
    stats = AsymExecutionStats()
    cfg = dict(MATRIX_1B_CONFIG if workload == "matrix_1b" else MM_CONFIGS[workload])
    cfg["tokens"] = requested_tokens(args, int(cfg["tokens"]))
    torch.manual_seed(101)
    weight = torch.randn(cfg["out_features"], cfg["in_features"], dtype=dtype)
    model = AsymFrozenLinear(
        weight,
        backend=args.backend,
        pin_memory=device.type == "cuda",
        stats=stats,
        precision=args.precision,
    )
    model.profile_name = "matrix"
    del weight

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(cfg["tokens"], cfg["in_features"], device=device, dtype=dtype, requires_grad=True)
        target = torch.randn(cfg["tokens"], cfg["out_features"], device=device, dtype=dtype)
        return x, target

    book = StageBook(device, timing_mode=args.timing_mode)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        with book.time("forward.matrix.base_frozen_asymgemm"):
            model.profile_name = "matrix"
            return model(x)

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return profiled_mse_loss(prediction.float(), target.float(), "loss.mse", book)

    run_fundamental_steps(book, make_batch, forward_fn, loss_fn, args, reset_stats=lambda: reset_execution_stats(stats))
    avg = average(book.values, args.measure_steps)
    return build_report(
        workload,
        model,
        device,
        avg,
        stats.as_dict(),
        forward_keys=["forward.matrix.base_frozen_asymgemm"],
        backward_group_prefixes=["backward.loss.mse", "backward.matrix.base_dx_asymgemm"],
        config=_fundamental_config(cfg),
    )


def profile_matrix_1b(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return profile_mm(args, device, dtype, "matrix_1b")


def profile_mlp_fundamental(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, workload: str) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymFrozenLinear

    class FundamentalMLP(torch.nn.Module):
        def __init__(self, stats: AsymExecutionStats) -> None:
            super().__init__()
            torch.manual_seed(202)
            self.fc1 = AsymFrozenLinear(
                torch.randn(cfg["hidden_features"], cfg["in_features"], dtype=dtype),
                backend=args.backend,
                pin_memory=device.type == "cuda",
                stats=stats,
                precision=args.precision,
            )
            self.fc2 = AsymFrozenLinear(
                torch.randn(cfg["out_features"], cfg["hidden_features"], dtype=dtype),
                backend=args.backend,
                pin_memory=device.type == "cuda",
                stats=stats,
                precision=args.precision,
            )

    clear(device)
    stats = AsymExecutionStats()
    cfg = dict(MLP_CONFIGS[workload])
    cfg["tokens"] = requested_tokens(args, int(cfg["tokens"]))
    model = FundamentalMLP(stats)

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(cfg["tokens"], cfg["in_features"], device=device, dtype=dtype, requires_grad=True)
        target = torch.randn(cfg["tokens"], cfg["out_features"], device=device, dtype=dtype)
        return x, target

    book = StageBook(device, timing_mode=args.timing_mode)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        with book.time("forward.fc1.base_frozen_asymgemm"):
            model.fc1.profile_name = "fc1"
            hidden = model.fc1(x)
        with book.time("forward.activation_relu"):
            hidden = profiled_relu(hidden, "activation_relu", book)
        with book.time("forward.fc2.base_frozen_asymgemm"):
            model.fc2.profile_name = "fc2"
            return model.fc2(hidden)

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return profiled_mse_loss(prediction.float(), target.float(), "loss.mse", book)

    run_fundamental_steps(book, make_batch, forward_fn, loss_fn, args, reset_stats=lambda: reset_execution_stats(stats))
    avg = average(book.values, args.measure_steps)
    return build_report(
        workload,
        model,
        device,
        avg,
        stats.as_dict(),
        forward_keys=[
            "forward.fc1.base_frozen_asymgemm",
            "forward.activation_relu",
            "forward.fc2.base_frozen_asymgemm",
        ],
        backward_group_prefixes=[
            "backward.loss.mse",
            "backward.fc2.base_dx_asymgemm",
            "backward.activation_relu",
            "backward.fc1.base_dx_asymgemm",
        ],
        config=_fundamental_config(cfg),
    )


def profile_mlp_1b(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return profile_mlp_fundamental(args, device, dtype, "mlp_1b")


def set_profile_names(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        setattr(module, "_m4_profile_name", name)
        if ".shared_experts." in name:
            layer = _layer_prefix_from_module_name(name)
            expert = name.rsplit(".", 1)[-1]
            prefix = f"forward.{layer}.shared_expert.{expert}" if layer else f"forward.shared_expert.{expert}"
            setattr(module, "_m4_profile_prefix", prefix)
        elif ".experts." in name:
            layer = _layer_prefix_from_module_name(name)
            expert = name.rsplit(".", 1)[-1]
            prefix = f"forward.{layer}.routed_expert.{expert}" if layer else f"forward.routed_expert.{expert}"
            setattr(module, "_m4_profile_prefix", prefix)


def _layer_prefix_from_module_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        return f"layers.{parts[1]}"
    return ""


def patch_dense_forward(book: StageBook) -> list[tuple[Any, str, Any]]:
    import asym_gemm.training.dense as dense

    originals: list[tuple[Any, str, Any]] = []

    def linear_prefix(module: Any) -> str:
        name = str(getattr(module, "_m4_profile_name", "linear"))
        layer = _layer_prefix_from_module_name(name)
        if ".self_attn." in name:
            proj = name.rsplit(".", 1)[-1]
            return f"forward.{layer}.attention.{proj}" if layer else f"forward.attention.{proj}"
        if ".mlp." in name:
            proj = name.rsplit(".", 1)[-1]
            return f"forward.{layer}.mlp.{proj}" if layer else f"forward.mlp.{proj}"
        return f"forward.{name}"

    original_linear = dense.AsymLoRALinear.forward

    def timed_linear(self: Any, x: torch.Tensor) -> torch.Tensor:
        prefix = linear_prefix(self)
        with book.time(f"{prefix}.base_frozen_asymgemm"):
            self.base_layer.profile_name = prefix.removeprefix("forward.")
            base = self.base_layer(x)
        with book.time(f"{prefix}.lora_A"):
            low_rank = profiled_linear(x.float(), self.lora_A["default"].weight, None, f"{prefix.removeprefix('forward.')}.lora_A", book)
        with book.time(f"{prefix}.lora_B"):
            lora_raw = profiled_linear(low_rank, self.lora_B["default"].weight, None, f"{prefix.removeprefix('forward.')}.lora_B", book)
        with book.time(f"{prefix}.add_cast_scale"):
            bprefix = prefix.removeprefix("forward.")
            lora = profiled_scale_cast(lora_raw, float(self.scaling), base.dtype, f"{bprefix}.add_cast_scale", book)
            return profiled_residual_add(base, lora, f"{bprefix}.base_lora_add", book)

    originals.append((dense.AsymLoRALinear, "forward", original_linear))
    dense.AsymLoRALinear.forward = timed_linear

    original_attn = dense.TinySelfAttention.forward

    def timed_attn(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        name = str(getattr(self, "_m4_profile_name", "attention"))
        layer = _layer_prefix_from_module_name(name)
        prefix = f"forward.{layer}.attention" if layer else "forward.attention"
        bprefix = prefix.removeprefix("forward.")
        batch, seq, hidden = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        with book.time(f"{prefix}.scores_matmul"):
            scores = profiled_matmul(q.float(), k.float().transpose(-2, -1), f"{bprefix}.scores_matmul", book) / (float(self.head_dim) ** 0.5)
        with book.time(f"{prefix}.causal_mask"):
            mask = torch.triu(torch.ones(seq, seq, device=hidden_states.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        with book.time(f"{prefix}.softmax"):
            probs = profiled_softmax(scores, -1, f"{bprefix}.softmax", book)
        with book.time(f"{prefix}.value_matmul"):
            context = profiled_matmul(probs, v.float(), f"{bprefix}.value_matmul", book).transpose(1, 2).contiguous().view(batch, seq, hidden)
        return self.o_proj(context.to(dtype=hidden_states.dtype))

    originals.append((dense.TinySelfAttention, "forward", original_attn))
    dense.TinySelfAttention.forward = timed_attn

    original_mlp = dense.TinyMLP.forward

    def timed_mlp(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        name = str(getattr(self, "_m4_profile_name", "mlp"))
        layer = _layer_prefix_from_module_name(name)
        prefix = f"forward.{layer}.mlp" if layer else "forward.mlp"
        bprefix = prefix.removeprefix("forward.")
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        with book.time(f"{prefix}.silu_mul_activation"):
            activated = profiled_silu_mul(gate, up, f"{bprefix}.silu_mul_activation", book)
        return self.down_proj(activated.to(dtype=hidden_states.dtype))

    originals.append((dense.TinyMLP, "forward", original_mlp))
    dense.TinyMLP.forward = timed_mlp

    original_layer = dense.TinyDecoderLayer.forward

    def timed_layer(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        name = str(getattr(self, "_m4_profile_name", "layer"))
        layer = f"forward.{name}" if _layer_prefix_from_module_name(name) else "forward"
        bprefix = layer.removeprefix("forward.")
        residual = hidden_states
        with book.time(f"{layer}.attention.layernorm"):
            hidden_states = profiled_layer_norm(self.input_layernorm, hidden_states, f"{bprefix}.attention.layernorm", book)
        attn_out = self.self_attn(hidden_states)
        with book.time(f"{layer}.attention.residual_add"):
            hidden_states = profiled_residual_add(residual, attn_out, f"{bprefix}.attention.residual_add", book)
        residual = hidden_states
        with book.time(f"{layer}.mlp.layernorm"):
            hidden_states = profiled_layer_norm(self.post_attention_layernorm, hidden_states, f"{bprefix}.mlp.layernorm", book)
        mlp_out = self.mlp(hidden_states)
        with book.time(f"{layer}.mlp.residual_add"):
            return profiled_residual_add(residual, mlp_out, f"{bprefix}.mlp.residual_add", book)

    originals.append((dense.TinyDecoderLayer, "forward", original_layer))
    dense.TinyDecoderLayer.forward = timed_layer

    original_base = dense.TinyDenseLLMBase.forward

    def timed_base(self: Any, *, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, labels: torch.Tensor | None = None, return_activations: bool = False) -> dict[str, Any]:
        with book.time("forward.embeddings"):
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = F.embedding(input_ids.to(device=self.embed_tokens_weight.device), self.embed_tokens_weight)
            else:
                hidden_states = inputs_embeds
            seq = hidden_states.shape[1]
            hidden_states = hidden_states + self.position_embedding[:seq].unsqueeze(0)
        activations = {}
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if return_activations:
                activations[f"layers.{index}"] = hidden_states
        with book.time("forward.final_norm"):
            hidden_states = profiled_layer_norm(self.final_layernorm, hidden_states, "final_norm", book)
        with book.time("forward.lm_head"):
            logits = profiled_linear(hidden_states, self.lm_head.weight, None, "lm_head", book)
        return {"logits": logits, "loss": None, "activations": activations}

    originals.append((dense.TinyDenseLLMBase, "forward", original_base))
    dense.TinyDenseLLMBase.forward = timed_base
    return originals


def dense_forward_keys(num_layers: int) -> list[str]:
    keys = ["forward.embeddings"]
    attention_ops = [
        "layernorm",
        "scores_matmul",
        "causal_mask",
        "softmax",
        "value_matmul",
        "residual_add",
    ]
    mlp_ops = ["layernorm", "silu_mul_activation", "residual_add"]
    for layer_idx in range(num_layers):
        layer = f"forward.layers.{layer_idx}"
        keys.extend(f"{layer}.attention.{op}" for op in attention_ops)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            keys.extend(
                [
                    f"{layer}.attention.{proj}.base_frozen_asymgemm",
                    f"{layer}.attention.{proj}.lora_A",
                    f"{layer}.attention.{proj}.lora_B",
                    f"{layer}.attention.{proj}.add_cast_scale",
                ]
            )
        keys.extend(f"{layer}.mlp.{op}" for op in mlp_ops)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.extend(
                [
                    f"{layer}.mlp.{proj}.base_frozen_asymgemm",
                    f"{layer}.mlp.{proj}.lora_A",
                    f"{layer}.mlp.{proj}.lora_B",
                    f"{layer}.mlp.{proj}.add_cast_scale",
                ]
            )
    keys.extend(["forward.final_norm", "forward.lm_head"])
    return keys


def dense_backward_prefixes(num_layers: int) -> list[str]:
    keys = ["backward.loss.cross_entropy", "backward.lm_head", "backward.final_norm"]
    for layer_idx in reversed(range(num_layers)):
        layer = f"backward.layers.{layer_idx}"
        keys.extend(
            [
                f"{layer}.mlp.residual_add",
                f"{layer}.mlp.down_proj.base_dx_asymgemm",
                f"{layer}.mlp.down_proj.base_lora_add",
                f"{layer}.mlp.down_proj.add_cast_scale",
                f"{layer}.mlp.down_proj.lora_B",
                f"{layer}.mlp.down_proj.lora_A",
                f"{layer}.mlp.silu_mul_activation",
                f"{layer}.mlp.up_proj.base_dx_asymgemm",
                f"{layer}.mlp.up_proj.base_lora_add",
                f"{layer}.mlp.up_proj.add_cast_scale",
                f"{layer}.mlp.up_proj.lora_B",
                f"{layer}.mlp.up_proj.lora_A",
                f"{layer}.mlp.gate_proj.base_dx_asymgemm",
                f"{layer}.mlp.gate_proj.base_lora_add",
                f"{layer}.mlp.gate_proj.add_cast_scale",
                f"{layer}.mlp.gate_proj.lora_B",
                f"{layer}.mlp.gate_proj.lora_A",
                f"{layer}.mlp.layernorm",
                f"{layer}.attention.residual_add",
                f"{layer}.attention.o_proj.base_dx_asymgemm",
                f"{layer}.attention.o_proj.base_lora_add",
                f"{layer}.attention.o_proj.add_cast_scale",
                f"{layer}.attention.o_proj.lora_B",
                f"{layer}.attention.o_proj.lora_A",
                f"{layer}.attention.value_matmul",
                f"{layer}.attention.softmax",
                f"{layer}.attention.scores_matmul",
                f"{layer}.attention.v_proj.base_dx_asymgemm",
                f"{layer}.attention.v_proj.base_lora_add",
                f"{layer}.attention.v_proj.add_cast_scale",
                f"{layer}.attention.v_proj.lora_B",
                f"{layer}.attention.v_proj.lora_A",
                f"{layer}.attention.k_proj.base_dx_asymgemm",
                f"{layer}.attention.k_proj.base_lora_add",
                f"{layer}.attention.k_proj.add_cast_scale",
                f"{layer}.attention.k_proj.lora_B",
                f"{layer}.attention.k_proj.lora_A",
                f"{layer}.attention.q_proj.base_dx_asymgemm",
                f"{layer}.attention.q_proj.base_lora_add",
                f"{layer}.attention.q_proj.add_cast_scale",
                f"{layer}.attention.q_proj.lora_B",
                f"{layer}.attention.q_proj.lora_A",
                f"{layer}.attention.layernorm",
            ]
        )
    return keys


def profile_dense(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.dense import AsymTinyDenseLLM, MICRO_DENSE_LLM_CONFIG, make_inputs, make_tiny_dense_weights

    clear(device)
    config = getattr(args, "_dense_config_override", MICRO_DENSE_LLM_CONFIG)
    if not hasattr(args, "_dense_config_override") and int(args.real_tokens or 0) > 0:
        config = replace(
            config,
            batch_size=int(args.real_batch_size),
            seq_len=int(args.real_seq_len),
            lora_rank=int(args.real_lora_rank),
            lora_alpha=float(args.real_lora_alpha),
        )
    workload_name = str(getattr(args, "_workload_name_override", "m4_2_dense_llm"))
    config_extra = dict(getattr(args, "_config_extra", {}))
    stats = AsymExecutionStats()
    weights = make_tiny_dense_weights(config, seed=1, dtype=dtype)
    model = AsymTinyDenseLLM(
        weights,
        config=config,
        target_mode="all",
        backend=args.backend,
        stats=stats,
        device=device,
        dtype=dtype,
        lora_seed=2,
        precision=args.precision,
    )
    set_profile_names(model)
    optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-3)

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = make_inputs(config, seed=int(time.perf_counter_ns() % 1_000_000), device=device, dtype=dtype)
        return inputs.detach().clone().requires_grad_(True), labels

    def forward_fn(model_: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        out = model_(inputs_embeds=inputs, labels=None)
        return out["logits"]

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous().to(device=logits.device)
        return profiled_cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), "loss.cross_entropy", book)

    book = StageBook(device, timing_mode=args.timing_mode)
    originals = patch_dense_forward(book)
    try:
        run_profile_steps(book, model, optimizer, make_batch, forward_fn, loss_fn, args, reset_stats=lambda: reset_execution_stats(stats))
    finally:
        restore(originals)
    avg = average(book.values, args.measure_steps)
    return build_report(
        workload_name,
        model,
        device,
        avg,
        stats.as_dict(),
        forward_keys=dense_forward_keys(config.num_layers),
        backward_group_prefixes=dense_backward_prefixes(config.num_layers),
        config={**asdict(config), **config_extra},
    )


def patch_moe_forward(book: StageBook) -> list[tuple[Any, str, Any]]:
    import asym_gemm.training.moe as moe

    originals: list[tuple[Any, str, Any]] = []

    original_attn = moe.TinySelfAttention.forward

    def timed_attn(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dim = hidden_states.dim()
        if original_dim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif original_dim != 3:
            raise ValueError(f"hidden_states must be [tokens, hidden] or [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        batch, seq, hidden = hidden_states.shape
        with book.time("forward.attention.q_proj_base"):
            q_raw = profiled_linear(hidden_states, self.q_proj.weight, None, "attention.q_proj_base", book)
        with book.time("forward.attention.k_proj_base"):
            k_raw = profiled_linear(hidden_states, self.k_proj.weight, None, "attention.k_proj_base", book)
        with book.time("forward.attention.v_proj_base"):
            v_raw = profiled_linear(hidden_states, self.v_proj.weight, None, "attention.v_proj_base", book)
        q = q_raw.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        with book.time("forward.attention.scores_matmul"):
            scores = profiled_matmul(q.float(), k.float().transpose(-2, -1), "attention.scores_matmul", book) / (float(self.head_dim) ** 0.5)
        with book.time("forward.attention.causal_mask"):
            mask = torch.triu(torch.ones(seq, seq, device=hidden_states.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        with book.time("forward.attention.softmax"):
            probs = profiled_softmax(scores, -1, "attention.softmax", book)
        with book.time("forward.attention.value_matmul"):
            context = profiled_matmul(probs, v.float(), "attention.value_matmul", book).transpose(1, 2).contiguous().view(batch, seq, hidden)
        with book.time("forward.attention.o_proj_base"):
            out = profiled_linear(context.to(dtype=hidden_states.dtype), self.o_proj.weight, None, "attention.o_proj_base", book)
        return out.squeeze(0) if original_dim == 2 else out

    originals.append((moe.TinySelfAttention, "forward", original_attn))
    moe.TinySelfAttention.forward = timed_attn

    original_expert = moe.AsymTinyExpert.forward

    def timed_expert(self: Any, x: torch.Tensor) -> torch.Tensor:
        prefix = str(getattr(self, "_m4_profile_prefix", "forward.routed_expert"))
        bprefix = prefix.removeprefix("forward.")

        def lora(name: str, value: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
            a = getattr(self, f"{name}_lora_a")
            b = getattr(self, f"{name}_lora_b")
            with book.time(f"{prefix}.{name}_lora_A"):
                low_rank = profiled_linear(value.float(), a, None, f"{bprefix}.{name}_lora_A", book)
            with book.time(f"{prefix}.{name}_lora_B"):
                out = profiled_linear(low_rank, b, None, f"{bprefix}.{name}_lora_B", book) * float(self.lora_scale)
            with book.time(f"{prefix}.{name}_lora_scale_cast"):
                return out.to(dtype=out_dtype)

        if x.numel() == 0:
            return x.new_empty((0, self.config.hidden_size))
        with book.time(f"{prefix}.contiguous_input"):
            x = x.contiguous()
        with book.time(f"{prefix}.gate_base_asymgemm"):
            self.gate_base.profile_name = f"{bprefix}.gate_base"
            gate_base = self.gate_base(x)
        gate_lora = lora("gate", x, x.dtype)
        gate = profiled_residual_add(gate_base, gate_lora, f"{bprefix}.gate_base_lora_add", book)
        with book.time(f"{prefix}.up_base_asymgemm"):
            self.up_base.profile_name = f"{bprefix}.up_base"
            up_base = self.up_base(x)
        up_lora = lora("up", x, x.dtype)
        up = profiled_residual_add(up_base, up_lora, f"{bprefix}.up_base_lora_add", book)
        with book.time(f"{prefix}.activation_silu_mul"):
            activated = profiled_silu_mul(gate, up, f"{bprefix}.activation_silu_mul", book).to(dtype=x.dtype)
        with book.time(f"{prefix}.down_base_asymgemm"):
            self.down_base.profile_name = f"{bprefix}.down_base"
            down_base = self.down_base(activated.contiguous())
        down_lora = lora("down", activated, x.dtype)
        with book.time(f"{prefix}.add"):
            return profiled_residual_add(down_base, down_lora, f"{bprefix}.down_base_lora_add", book)

    originals.append((moe.AsymTinyExpert, "forward", original_expert))
    moe.AsymTinyExpert.forward = timed_expert

    original_layer_forward = moe.AsymTinyMoELayer.forward

    def timed_layer_forward(self: Any, x: torch.Tensor, *, static_routing: Any = None, mode: str = "contiguous", return_details: bool = False) -> Any:
        residual = x
        with book.time("forward.attention.layernorm"):
            attn_in = profiled_frozen_layer_norm(self.input_layernorm, x, "attention.layernorm", book)
        attn_out = self.self_attn(attn_in)
        with book.time("forward.attention.residual_add"):
            hidden = profiled_residual_add(residual, attn_out, "attention.residual_add", book)
        with book.time("forward.moe.layernorm"):
            moe_in = profiled_frozen_layer_norm(self.post_attention_layernorm, hidden, "moe.layernorm", book)
        moe_out, details = self._run_moe(moe_in, static_routing=static_routing, mode=mode)
        with book.time("forward.moe.residual_add"):
            next_x = profiled_residual_add(hidden, moe_out, "moe.residual_add", book, scale=float(self.config.residual_scale))
        if not return_details:
            return next_x
        return next_x, details

    originals.append((moe.AsymTinyMoELayer, "forward", original_layer_forward))
    moe.AsymTinyMoELayer.forward = timed_layer_forward

    original_run_moe = moe.AsymTinyMoELayer._run_moe

    def timed_run_moe(self: Any, x: torch.Tensor, *, static_routing: Any, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
        input_shape = x.shape
        with book.time("forward.moe.flatten"):
            flat = x.reshape(-1, self.config.hidden_size)
        with book.time("forward.router"):
            (topk_indices, routing_weights), logits = self._route(flat, static_routing)
        with book.time("forward.route_metadata"):
            metadata = moe.build_route_metadata(topk_indices, routing_weights, num_experts=self.config.num_experts, mode=mode)
        if mode == "contiguous":
            with book.time("forward.pack_tokens"):
                packed = profiled_pack_tokens(flat, metadata, "contiguous", book)
            with book.time("forward.routed_expert.dispatch_loop"):
                expert_output = self._run_contiguous(packed, metadata)
            with book.time("forward.scatter_combine"):
                routed_out = profiled_scatter_tokens(expert_output, metadata, "contiguous", book)
        else:
            with book.time("forward.pack_tokens"):
                packed = profiled_pack_tokens(flat, metadata, "masked", book)
            with book.time("forward.routed_expert.dispatch_loop"):
                expert_output = self._run_masked(packed, metadata)
            with book.time("forward.scatter_combine"):
                routed_out = profiled_scatter_tokens(expert_output, metadata, "masked", book)
        with book.time("forward.shared_expert.dispatch_loop"):
            shared = self._run_shared(flat)
        with book.time("forward.moe.combine_shared_routed"):
            moe_out = profiled_residual_add(routed_out, shared, "moe.combine_shared_routed", book)
        return moe_out.reshape(input_shape), {
            "metadata": metadata,
            "logits": logits,
            "topk_indices": topk_indices,
            "routing_weights": routing_weights,
        }

    originals.append((moe.AsymTinyMoELayer, "_run_moe", original_run_moe))
    moe.AsymTinyMoELayer._run_moe = timed_run_moe

    original_model_forward = moe.TinyMoE.forward

    def timed_model_forward(self: Any, x: torch.Tensor | None = None, *, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, labels: torch.Tensor | None = None, static_routing: Any = None, mode: str = "contiguous", return_details: bool = False) -> Any:
        details = []
        with book.time("forward.embeddings"):
            hidden, token_api = self._prepare_hidden(x, input_ids, inputs_embeds)
        for layer_idx, layer in enumerate(self.layers):
            result = layer(hidden, static_routing=moe._routing_for_layer(static_routing, layer_idx), mode=mode, return_details=return_details)
            if return_details:
                hidden, detail = result
                details.append(detail)
            else:
                hidden = result
        with book.time("forward.final_norm"):
            hidden = profiled_frozen_layer_norm(self.final_layernorm, hidden, "final_norm", book)
        if token_api or labels is not None:
            with book.time("forward.lm_head"):
                logits = self.lm_head(hidden)
            return {"logits": logits, "loss": None, "hidden_states": hidden, "details": details if return_details else []}
        if return_details:
            return hidden, details
        return hidden

    originals.append((moe.TinyMoE, "forward", original_model_forward))
    moe.TinyMoE.forward = timed_model_forward
    return originals


def moe_forward_keys(config: Any) -> list[str]:
    keys = [
        "forward.embeddings",
        "forward.attention.layernorm",
        "forward.attention.q_proj_base",
        "forward.attention.k_proj_base",
        "forward.attention.v_proj_base",
        "forward.attention.scores_matmul",
        "forward.attention.causal_mask",
        "forward.attention.softmax",
        "forward.attention.value_matmul",
        "forward.attention.o_proj_base",
        "forward.attention.residual_add",
        "forward.moe.layernorm",
        "forward.moe.flatten",
        "forward.router",
        "forward.route_metadata",
        "forward.pack_tokens",
        "forward.scatter_combine",
    ]
    expert_ops = [
        "contiguous_input",
        "gate_base_asymgemm",
        "gate_lora_A",
        "gate_lora_B",
        "gate_lora_scale_cast",
        "up_base_asymgemm",
        "up_lora_A",
        "up_lora_B",
        "up_lora_scale_cast",
        "activation_silu_mul",
        "down_base_asymgemm",
        "down_lora_A",
        "down_lora_B",
        "down_lora_scale_cast",
        "add",
    ]
    for layer_idx in range(config.num_layers):
        for expert_idx in range(config.num_experts):
            keys.extend(f"forward.layers.{layer_idx}.routed_expert.{expert_idx}.{op}" for op in expert_ops)
        for expert_idx in range(config.num_shared_experts):
            keys.extend(f"forward.layers.{layer_idx}.shared_expert.{expert_idx}.{op}" for op in expert_ops)
    keys.extend(["forward.moe.combine_shared_routed", "forward.moe.residual_add", "forward.final_norm"])
    return keys


def moe_backward_prefixes(config: Any) -> list[str]:
    keys = [
        "backward.loss.mse",
        "backward.final_norm",
        "backward.moe.residual_add",
        "backward.moe.combine_shared_routed",
        "backward.scatter_combine",
        "backward.pack_tokens",
        "backward.route_metadata",
        "backward.router",
    ]
    expert_ops = [
        "down_base.base_dx_asymgemm",
        "down_base_lora_add",
        "down_lora_B",
        "down_lora_A",
        "activation_silu_mul",
        "up_base.base_dx_asymgemm",
        "up_base_lora_add",
        "up_lora_B",
        "up_lora_A",
        "gate_base.base_dx_asymgemm",
        "gate_base_lora_add",
        "gate_lora_B",
        "gate_lora_A",
    ]
    for layer_idx in reversed(range(config.num_layers)):
        for expert_idx in range(config.num_experts):
            keys.extend(f"backward.layers.{layer_idx}.routed_expert.{expert_idx}.{op}" for op in expert_ops)
        for expert_idx in range(config.num_shared_experts):
            keys.extend(f"backward.layers.{layer_idx}.shared_expert.{expert_idx}.{op}" for op in expert_ops)
    keys.extend(
        [
            "backward.moe.layernorm",
            "backward.attention.residual_add",
            "backward.attention.o_proj_base",
            "backward.attention.value_matmul",
            "backward.attention.softmax",
            "backward.attention.scores_matmul",
            "backward.attention.v_proj_base",
            "backward.attention.k_proj_base",
            "backward.attention.q_proj_base",
            "backward.attention.layernorm",
        ]
    )
    return keys


def profile_moe(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.moe import MICRO_MOE_CONFIG, make_static_routes, make_tiny_moe_pair

    clear(device)
    config = getattr(args, "_moe_config_override", MICRO_MOE_CONFIG)
    if not hasattr(args, "_moe_config_override") and int(args.real_tokens or 0) > 0:
        config = replace(
            config,
            batch_size=int(args.real_batch_size),
            seq_len=int(args.real_seq_len),
            logical_tokens=requested_tokens(args, int(config.logical_tokens)),
            lora_rank=int(args.real_lora_rank),
            lora_alpha=float(args.real_lora_alpha),
        )
    workload_name = str(getattr(args, "_workload_name_override", "m4_3_moe"))
    config_extra = dict(getattr(args, "_config_extra", {}))
    model, _, _, stats = make_tiny_moe_pair(
        config=config,
        seed=3,
        device=device,
        base_dtype=dtype,
        backend=args.backend,
        pin_memory=device.type == "cuda",
        precision=args.precision,
    )
    set_profile_names(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    static_routes = make_static_routes(config, device, pattern="balanced")

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(config.logical_tokens, config.hidden_size, device=device, dtype=dtype, requires_grad=True) * 0.5
        target = torch.roll(x.detach().float(), shifts=1, dims=0) * 0.25
        return x, target

    def forward_fn(model_: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = model_(x, static_routing=static_routes, mode=args.moe_mode)
        assert isinstance(y, torch.Tensor)
        return y

    book = StageBook(device, timing_mode=args.timing_mode)
    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return profiled_mse_loss(prediction.float(), target, "loss.mse", book)

    originals = patch_moe_forward(book)
    try:
        run_profile_steps(book, model, optimizer, make_batch, forward_fn, loss_fn, args, reset_stats=lambda: reset_execution_stats(stats))
    finally:
        restore(originals)
    avg = average(book.values, args.measure_steps)
    return build_report(
        workload_name,
        model,
        device,
        avg,
        stats.as_dict(),
        forward_keys=moe_forward_keys(config),
        backward_group_prefixes=moe_backward_prefixes(config),
        config={**asdict(config), "moe_mode": args.moe_mode, **config_extra},
    )


def run_profile_steps(
    book: StageBook,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    make_batch: Callable[[], tuple[Any, Any]],
    forward_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Any], torch.Tensor],
    args: argparse.Namespace,
    reset_stats: Callable[[], None] | None = None,
) -> None:
    with profile_enabled(True):
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            batch, target = make_batch()
            loss = loss_fn(forward_fn(model, batch), target)
            loss.backward()
            optimizer.step()
            sync(book.device)
        if reset_stats is not None:
            reset_stats()
        book.values.clear()

        with patch_base_dispatch(book):
            for _ in range(args.measure_steps):
                with book.time("step.input_preparation"):
                    batch, target = make_batch()
                optimizer.zero_grad(set_to_none=True)
                with book.time("step.forward"):
                    prediction = forward_fn(model, batch)
                with book.time("step.loss"):
                    loss = loss_fn(prediction, target)
                with book.time("step.backward"):
                    loss.backward()
                with book.time("step.optimizer"):
                    optimizer.step()
                sync(book.device)
                book.flush()


def build_report(
    workload: str,
    model: torch.nn.Module,
    device: torch.device,
    avg: dict[str, float],
    execution_stats: dict[str, Any],
    *,
    forward_keys: list[str],
    backward_group_prefixes: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    forward_total = avg.get("step.forward", 0.0)
    backward_total = avg.get("step.backward", 0.0)
    step_total = sum(avg.get(key, 0.0) for key in ("step.input_preparation", "step.forward", "step.loss", "step.backward", "step.optimizer"))
    backward_keys: list[str] = []
    for prefix in backward_group_prefixes:
        value = sum(v for k, v in avg.items() if k.startswith(prefix))
        key = prefix + ".total"
        avg[key] = value
        backward_keys.append(key)
    avg["backward.python_autograd_dispatch_cuda_launch_and_sync"] = max(0.0, backward_total - sum(avg.get(key, 0.0) for key in backward_keys))
    backward_keys.append("backward.python_autograd_dispatch_cuda_launch_and_sync")

    return {
        "workload": workload,
        "device": str(device),
        "asym_precision_requested": str(config.get("asym_precision_requested", getattr(model, "precision", "bf16"))),
        "asym_precision_effective": str(getattr(model, "precision", "bf16")),
        "config": config,
        "execution_stats": execution_stats,
        "step": table(
            step_total,
            rows_from_keys(avg, step_total, ["step.input_preparation", "step.forward", "step.loss", "step.backward", "step.optimizer"], residual_name="step.accounting_gap"),
        ),
        "forward": table(forward_total, rows_from_keys(avg, forward_total, forward_keys, residual_name="forward.python_dispatch_cuda_launch_and_sync")),
        "backward": table(backward_total, rows_from_keys(avg, backward_total, backward_keys, residual_name="backward.accounting_gap")),
        "memory": memory_report(model, device),
        "raw_seconds_per_step": raw_seconds_without_individual_calls(avg),
        "raw_dispatch_call_group_seconds_per_step": call_group_seconds_per_step(avg),
        "notes": [
            "Source-level rows are Python wall-clock range timings. In timing_mode=profile, inner ranges do not synchronize and should be treated as labels/CPU submission timing, not GPU execution timing.",
            "Use timing_mode=profile under Nsight Systems for real GPU bubble analysis; inner ranges are NVTX/record_function labels and do not force per-op CUDA synchronization.",
            "timing_mode=debug_sync is a debugging-only source coverage check that synchronizes every region and must not be used for performance claims.",
            "All explicit tensor ops in the instrumented toy forward/backward are assigned named ranges. Source-level residual rows are non-tensor-op overhead: Python dispatch, PyTorch autograd engine scheduling, CUDA launch latency, and synchronization/profiler overhead.",
            "pinned_w_t_bytes is an audit field and should remain zero after transpose_b=True dX.",
            "The optional torch_profiler_backward table is a debug aid averaged over all profiled steps, including warmup; Nsight tables remain the performance truth.",
        ],
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report["workload"]
    json_path = output_dir / f"{stem}_profile.json"
    md_path = output_dir / f"{stem}_profile.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    moe_mode = report.get("config", {}).get("moe_mode")
    if report["workload"] == "m4_3_moe" and moe_mode:
        mode_stem = f"m4_3_moe_{moe_mode}"
        (output_dir / f"{mode_stem}_profile.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / f"{mode_stem}_profile.md").write_text(markdown(report), encoding="utf-8")


def markdown(report: dict[str, Any]) -> str:
    def emit_table(title: str, section: dict[str, Any]) -> list[str]:
        lines = [f"## {title}", "", "| Component | ms | % |", "|---|---:|---:|"]
        for row in section["rows"]:
            lines.append(f"| {row['name']} | {row['milliseconds']:.4f} | {row['percent']:.2f}% |")
        lines.append(f"| **sum** | {section['sum_seconds'] * 1000.0:.4f} | {section['sum_percent']:.2f}% |")
        lines.append("")
        return lines

    mem = report["memory"]
    lines = [
        f"# {report['workload']} Source-Label Coverage Report",
        "",
        "This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_m4.py` for kernel-busy, memcpy, and GPU no-kernel percentages.",
        "",
        "These source-label tables are for auditing range coverage. In `--timing-mode profile`, inner rows are asynchronous Python/NVTX range timings. In `--timing-mode debug_sync`, rows are synchronized debugging timings and carry profiler overhead.",
        "",
    ]
    lines.extend(emit_table("Step", report["step"]))
    lines.extend(emit_table("Forward", report["forward"]))
    lines.extend(emit_table("Backward", report["backward"]))
    if "torch_profiler_backward" in report:
        lines.extend(emit_table("Torch Profiler Backward CUDA Debug Avg", report["torch_profiler_backward"]))
    lines.extend(
        [
            "## Memory",
            "",
            "| Component | bytes |",
            "|---|---:|",
            f"| peak_hbm | {mem['gpu']['peak_hbm_bytes']} |",
            f"| gpu_parameters | {mem['gpu']['parameter_bytes']} |",
            f"| gpu_buffers | {mem['gpu']['buffer_bytes']} |",
            f"| host_W | {mem['cpu']['host_w_bytes']} |",
            f"| host_W_T | {mem['cpu']['host_w_t_bytes']} |",
            f"| pinned_W | {mem['cpu']['pinned_w_bytes']} |",
            f"| pinned_W_T | {mem['cpu']['pinned_w_t_bytes']} |",
            f"| pinned_total | {mem['cpu']['pinned_total_bytes']} |",
            "",
        ]
    )
    return "\n".join(lines)


def _event_device_seconds(event: Any) -> float:
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0.0)
    return float(value or 0.0) / 1_000_000.0


def _event_cpu_seconds(event: Any) -> float:
    return float(getattr(event, "cpu_time_total", 0.0) or 0.0) / 1_000_000.0


def _has_parent(event: Any, parent_name: str) -> bool:
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        if getattr(parent, "name", "") == parent_name:
            return True
        parent = getattr(parent, "cpu_parent", None)
    return False


def _classify_backward_event(name: str) -> str:
    lower = name.lower()
    if lower.startswith("backward.loss."):
        return "backward.loss_grad"
    if "base_dx_asymgemm" in lower:
        return "backward.base_dx_asymgemm"
    if ".lora_a." in lower or "_lora_a." in lower:
        return "backward.lora_A_grad"
    if ".lora_b." in lower or "_lora_b." in lower:
        return "backward.lora_B_grad"
    if "residual_add" in lower or "base_lora_add" in lower or "add_cast_scale" in lower:
        return "backward.add_scale_cast_grad"
    if "softmaxbackward" in lower or "_softmax_backward" in lower:
        return "backward.softmax_grad"
    if "relubackward" in lower or "silu" in lower or "sigmoidbackward" in lower or "threshold_backward" in lower:
        return "backward.activation_grad"
    if "layernormbackward" in lower or "native_layer_norm_backward" in lower:
        return "backward.layernorm_grad"
    if "indexadd" in lower or "scatter" in lower or "gather" in lower or "index_select_backward" in lower:
        return "backward.route_scatter_pack_grad"
    if "mmbackward" in lower or "addmmbackward" in lower or lower in {"aten::mm", "aten::bmm", "aten::addmm"}:
        return "backward.matmul_or_lora_grad"
    if "autograd" in lower or "evaluate_function" in lower:
        return "backward.autograd_engine"
    return "backward.autograd_other"


def profiler_backward_table(prof: Any, *, profiled_steps: int) -> dict[str, Any]:
    values: dict[str, float] = defaultdict(float)
    for event in prof.events():
        name = str(getattr(event, "name", ""))
        if not name.startswith("backward.") and not _has_parent(event, "step.backward") and name != "step.backward":
            continue
        seconds = _event_cpu_seconds(event) if name.startswith("backward.") else _event_device_seconds(event)
        if seconds <= 0.0:
            continue
        values[_classify_backward_event(name)] += seconds
    divisor = float(max(1, profiled_steps))
    values = defaultdict(float, {key: value / divisor for key, value in values.items()})
    total = sum(values.values())
    preferred = [
        "backward.loss_grad",
        "backward.base_dx_asymgemm",
        "backward.lora_B_grad",
        "backward.lora_A_grad",
        "backward.add_scale_cast_grad",
        "backward.activation_grad",
        "backward.softmax_grad",
        "backward.layernorm_grad",
        "backward.route_scatter_pack_grad",
        "backward.matmul_or_lora_grad",
        "backward.autograd_engine",
        "backward.autograd_other",
    ]
    return table(total, rows_from_keys(values, total, preferred))


def qwen3_14b_dense_config(args: argparse.Namespace) -> Any:
    from asym_gemm.training.dense import TinyDenseLLMConfig

    full = QWEN3_14B_CONFIG
    return TinyDenseLLMConfig(
        vocab_size=min(int(full["vocab_size"]), int(args.real_vocab_rows)),
        hidden_size=int(full["hidden_size"]),
        num_layers=max(1, min(int(args.real_profile_layers), int(full["hf_num_hidden_layers"]))),
        num_heads=int(full["num_attention_heads"]),
        seq_len=int(args.real_seq_len),
        batch_size=int(args.real_batch_size),
        intermediate_size=int(full["intermediate_size"]),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
    )


def custom_dense_3b_config(args: argparse.Namespace) -> Any:
    from asym_gemm.training.dense import TinyDenseLLMConfig

    full = CUSTOM_DENSE_3B_CONFIG
    return TinyDenseLLMConfig(
        vocab_size=min(int(full["vocab_size"]), int(args.real_vocab_rows)),
        hidden_size=int(full["hidden_size"]),
        num_layers=max(1, min(int(args.real_profile_layers), int(full["hf_num_hidden_layers"]))),
        num_heads=int(full["num_attention_heads"]),
        seq_len=int(args.real_seq_len),
        batch_size=int(args.real_batch_size),
        intermediate_size=int(full["intermediate_size"]),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
    )


def qwen3_30b_a3b_moe_config(args: argparse.Namespace) -> Any:
    from asym_gemm.training.moe import TinyMoEConfig

    full = QWEN3_30B_A3B_CONFIG
    return TinyMoEConfig(
        num_layers=max(1, min(int(args.real_profile_layers), int(full["hf_num_hidden_layers"]))),
        num_experts=int(full["num_experts"]),
        top_k=int(full["num_experts_per_tok"]),
        hidden_size=int(full["hidden_size"]),
        intermediate_size=int(full["moe_intermediate_size"]),
        logical_tokens=int(args.real_tokens or (int(args.real_seq_len) * int(args.real_batch_size))),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
        residual_scale=0.25,
        num_shared_experts=int(full["num_shared_experts"]),
        vocab_size=min(int(full["vocab_size"]), int(args.real_vocab_rows)),
        num_heads=int(full["num_attention_heads"]),
        batch_size=int(args.real_batch_size),
        seq_len=int(args.real_seq_len),
    )


def custom_moe_3b_config(args: argparse.Namespace) -> Any:
    from asym_gemm.training.moe import TinyMoEConfig

    full = CUSTOM_MOE_3B_CONFIG
    return TinyMoEConfig(
        num_layers=max(1, min(int(args.real_profile_layers), int(full["hf_num_hidden_layers"]))),
        num_experts=int(full["num_experts"]),
        top_k=int(full["num_experts_per_tok"]),
        hidden_size=int(full["hidden_size"]),
        intermediate_size=int(full["moe_intermediate_size"]),
        logical_tokens=int(args.real_tokens or (int(args.real_seq_len) * int(args.real_batch_size))),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
        residual_scale=0.25,
        num_shared_experts=int(full["num_shared_experts"]),
        vocab_size=min(int(full["vocab_size"]), int(args.real_vocab_rows)),
        num_heads=int(full["num_attention_heads"]),
        batch_size=int(args.real_batch_size),
        seq_len=int(args.real_seq_len),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=["mlp", "dense", "moe", "dense_3b", "moe_3b", "qwen3_14b", "qwen3_30b_a3b", "matrix_1b", "mm_1b", "mm_3b", "mlp_1b", "mlp_3b"],
        default="mlp",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backend", choices=["asym_only", "asym_or_staged", "asym_or_torch", "torch_only"], default="asym_only")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
    parser.add_argument("--profile-layers", "--real-profile-layers", dest="real_profile_layers", metavar="N", type=int, default=1)
    parser.add_argument("--batch-size", "--real-batch-size", dest="real_batch_size", metavar="N", type=int, default=1)
    parser.add_argument("--seq-len", "--real-seq-len", dest="real_seq_len", metavar="N", type=int, default=64)
    parser.add_argument("--tokens", "--real-tokens", dest="real_tokens", metavar="N", type=int, default=0)
    parser.add_argument("--lora-rank", "--real-lora-rank", dest="real_lora_rank", metavar="N", type=int, default=64)
    parser.add_argument("--lora-alpha", "--real-lora-alpha", dest="real_lora_alpha", metavar="FLOAT", type=float, default=128.0)
    parser.add_argument("--vocab-rows", "--real-vocab-rows", dest="real_vocab_rows", metavar="N", type=int, default=4096)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "fp4"])
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--export-torch-trace", action="store_true", help="Export a PyTorch profiler Chrome trace for the measured run")
    parser.add_argument(
        "--torch-profiler-with-stack",
        "--torch-profiler-stack-debug",
        dest="torch_profiler_with_stack",
        action="store_true",
        help="Enable PyTorch profiler stack capture when exporting a trace. This is debug-only and adds noticeable overhead.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["profile", "debug_sync"],
        default="profile",
        help="profile is the real low-overhead NVTX/Nsight mode; debug_sync synchronizes every range for source coverage debugging only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    def run() -> dict[str, Any]:
        if args.workload == "mlp":
            return profile_mlp(args, device, dtype)
        if args.workload == "dense":
            return profile_dense(args, device, dtype)
        if args.workload == "moe":
            return profile_moe(args, device, dtype)
        if args.workload == "dense_3b":
            args._dense_config_override = custom_dense_3b_config(args)
            args._workload_name_override = "dense_3b"
            args._config_extra = {
                **CUSTOM_DENSE_3B_CONFIG,
                "profiled_layers": args._dense_config_override.num_layers,
                "weight_source": "random_config_matched",
            }
            return profile_dense(args, device, dtype)
        if args.workload == "moe_3b":
            args._moe_config_override = custom_moe_3b_config(args)
            args._workload_name_override = "moe_3b"
            args._config_extra = {
                **CUSTOM_MOE_3B_CONFIG,
                "profiled_layers": args._moe_config_override.num_layers,
                "weight_source": "random_config_matched",
            }
            return profile_moe(args, device, dtype)
        if args.workload == "matrix_1b":
            return profile_matrix_1b(args, device, dtype)
        if args.workload in {"mm_1b", "mm_3b"}:
            return profile_mm(args, device, dtype, args.workload)
        if args.workload == "mlp_1b":
            return profile_mlp_1b(args, device, dtype)
        if args.workload == "mlp_3b":
            return profile_mlp_fundamental(args, device, dtype, "mlp_3b")
        if args.workload == "qwen3_14b":
            args._dense_config_override = qwen3_14b_dense_config(args)
            args._workload_name_override = "qwen3_14b"
            args._config_extra = {
                **QWEN3_14B_CONFIG,
                "profiled_layers": args._dense_config_override.num_layers,
                "weight_source": "random_config_matched",
            }
            return profile_dense(args, device, dtype)
        if args.workload == "qwen3_30b_a3b":
            args._moe_config_override = qwen3_30b_a3b_moe_config(args)
            args._workload_name_override = "qwen3_30b_a3b"
            args._config_extra = {
                **QWEN3_30B_A3B_CONFIG,
                "profiled_layers": args._moe_config_override.num_layers,
                "weight_source": "random_config_matched",
            }
            return profile_moe(args, device, dtype)
        raise AssertionError(args.workload)

    if args.export_torch_trace:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=args.torch_profiler_with_stack,
            with_modules=True,
        ) as prof:
            report = run()
        report["torch_profiler_backward"] = profiler_backward_table(prof, profiled_steps=args.warmup_steps + args.measure_steps)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.output_dir / f"{report['workload']}_torch_trace.json"))
    else:
        report = run()
    write_report(report, args.output_dir)
    print(json.dumps({"workload": report["workload"], "step_ms": report["step"]["total_milliseconds"]}, indent=2))


if __name__ == "__main__":
    main()
