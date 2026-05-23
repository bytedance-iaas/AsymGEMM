#!/usr/bin/env python3
"""LoRA-SFT profiling reports.

This profiler intentionally reports additive tables.  Rows that cannot be
split safely from PyTorch autograd are kept in explicit *_other buckets.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import gc
import json
import re
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
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asym_gemm.training.kt_moe import KTBackendUnavailable
from asym_gemm.training.profile_ranges import current_profile_range, profile_enabled


DENSE_14B_CONFIG = {
    "hf_model_id": "Qwen/Qwen3-14B",
    "hf_model_type": "qwen3",
    "hf_num_hidden_layers": 40,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 40,
    "vocab_size": 151936,
}


# MoE public workload names describe one routed-expert layer, not the full
# reference model depth: moe-<total routed expert params>m-a<active routed
# expert params per token>m. Layer count is controlled separately by
# --profile-layers or driver workload specs such as moe-604m-a75m|2.
MOE_604M_A38M_CONFIG = {
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


MOE_604M_A75M_CONFIG = {
    "hf_model_id": "custom/moe-3b-active",
    "hf_model_type": "moe-604m-a75m",
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


DEFAULT_LORA_BATCH_SIZE = 32
DEFAULT_LORA_SEQ_LEN = 64
DEFAULT_LORA_HIDDEN_DIM = 1024
DEFAULT_LORA_MLP_EXPANSION = 4


MM_CONFIGS = {
    "mm_1b": {
        "tokens": 64,
        "in_features": 32768,
        "out_features": 32768,
    },
    "mm_3b": {
        "batch_size": DEFAULT_LORA_BATCH_SIZE,
        "seq_len": DEFAULT_LORA_SEQ_LEN,
        "tokens": DEFAULT_LORA_BATCH_SIZE * DEFAULT_LORA_SEQ_LEN,
        "hidden_dim": DEFAULT_LORA_HIDDEN_DIM,
        "in_features": DEFAULT_LORA_HIDDEN_DIM,
        "out_features": DEFAULT_LORA_HIDDEN_DIM,
    },
}


MLP_CONFIGS = {
    "mlp_1b": {
        "tokens": 64,
        "in_features": 8192,
        "hidden_features": 65536,
        "out_features": 8192,
    },
    "mlp_3b": {
        "batch_size": DEFAULT_LORA_BATCH_SIZE,
        "seq_len": DEFAULT_LORA_SEQ_LEN,
        "tokens": DEFAULT_LORA_BATCH_SIZE * DEFAULT_LORA_SEQ_LEN,
        "hidden_dim": DEFAULT_LORA_HIDDEN_DIM,
        "in_features": DEFAULT_LORA_HIDDEN_DIM,
        "hidden_features": DEFAULT_LORA_HIDDEN_DIM * DEFAULT_LORA_MLP_EXPANSION,
        "out_features": DEFAULT_LORA_HIDDEN_DIM,
        "mlp_expansion": DEFAULT_LORA_MLP_EXPANSION,
    },
}
MLP_1B_CONFIG = MLP_CONFIGS["mlp_1b"]


WORKLOAD_CHOICES = (
    "mlp",
    "dense",
    "moe",
    "dense_3b",
    "dense_14b",
    "moe-604m-a75m",
    "moe-604m-a38m",
    "mm_1b",
    "mm_3b",
    "mlp_1b",
    "mlp_3b",
)
BACKEND_CHOICES = ("torch", "asym", "kt")
KT_MOE_WORKLOADS = ("moe", "moe-604m-a75m", "moe-604m-a38m")

LORA_FORWARD_OPS = ("base_frozen_asymgemm", "lora_A", "lora_B", "add_cast_scale")
LORA_BACKWARD_OPS = ("base_dx_asymgemm", "base_lora_add", "add_cast_scale", "lora_B", "lora_A")
DENSE_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
DENSE_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DENSE_ALL_PROJECTIONS = DENSE_ATTENTION_PROJECTIONS + DENSE_MLP_PROJECTIONS
DENSE_TARGET_MODES = ("mlp_only", "attention_only", "all")
DEFAULT_DENSE_TARGET_MODE = "all"
DEFAULT_TARGET_MODULES = "all"
DEFAULT_DENSE_OFFLOAD_MODULES = "mlp"
DEFAULT_MOE_OFFLOAD_MODULES = "routed_experts"
LORA_DTYPE_CHOICES = ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")
KT_LORA_DTYPE_CHOICES = ("bf16", "bfloat16")
KT_METHOD_CHOICES = ("AMXBF16_SFT", "AMXINT8_SFT", "AMXINT4_SFT")


def is_torch_backend(backend: str) -> bool:
    return backend == "torch"


def is_asym_backend(backend: str) -> bool:
    return backend == "asym"


def is_kt_backend(backend: str) -> bool:
    return backend == "kt"


def validate_backend_workload(args: argparse.Namespace) -> None:
    if is_kt_backend(args.backend) and args.workload not in KT_MOE_WORKLOADS:
        raise ValueError("backend=kt is only implemented for MoE LoRA SFT workloads.")
    if is_kt_backend(args.backend):
        lora_dtype = str(getattr(args, "lora_dtype", "bf16")).lower()
        if lora_dtype not in KT_LORA_DTYPE_CHOICES:
            raise ValueError("backend=kt currently supports BF16 LoRA buffers only.")


@dataclass(frozen=True)
class LoRALinearProfile:
    prefix: str

    def forward_keys(self) -> list[str]:
        return [f"forward.{self.prefix}.{op}" for op in LORA_FORWARD_OPS]

    def backward_prefixes(self) -> list[str]:
        return [f"backward.{self.prefix}.{op}" for op in LORA_BACKWARD_OPS]


@dataclass(frozen=True)
class RegressionBatchShape:
    tokens: int
    in_features: int
    out_features: int
    batch_size: int = 0
    seq_len: int = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RegressionBatchShape":
        return cls(
            tokens=int(config["tokens"]),
            in_features=int(config["in_features"]),
            out_features=int(config["out_features"]),
            batch_size=int(config.get("batch_size", 0) or 0),
            seq_len=int(config.get("seq_len", 0) or 0),
        )

    def input_shape(self) -> tuple[int, ...]:
        if self.batch_size > 0 and self.seq_len > 0 and self.batch_size * self.seq_len == self.tokens:
            return (self.batch_size, self.seq_len, self.in_features)
        return (self.tokens, self.in_features)

    def target_shape(self) -> tuple[int, ...]:
        if self.batch_size > 0 and self.seq_len > 0 and self.batch_size * self.seq_len == self.tokens:
            return (self.batch_size, self.seq_len, self.out_features)
        return (self.tokens, self.out_features)


@dataclass(frozen=True)
class LoRAAdapterTensors:
    a: torch.Tensor
    b: torch.Tensor
    scaling: float


def lora_forward_keys(*prefixes: str) -> list[str]:
    keys: list[str] = []
    for prefix in prefixes:
        keys.extend(LoRALinearProfile(prefix).forward_keys())
    return keys


def lora_backward_prefixes(*prefixes: str) -> list[str]:
    keys: list[str] = []
    for prefix in prefixes:
        keys.extend(LoRALinearProfile(prefix).backward_prefixes())
    return keys


def mlp_lora_forward_keys(*, activation_prefix: str = "activation_relu") -> list[str]:
    return [
        *lora_forward_keys("fc1"),
        f"forward.{activation_prefix}",
        *lora_forward_keys("fc2"),
    ]


def mlp_lora_backward_prefixes(
    *,
    loss_prefix: str = "loss.mse",
    activation_prefix: str = "activation_relu",
) -> list[str]:
    return [
        f"backward.{loss_prefix}",
        *lora_backward_prefixes("fc2"),
        f"backward.{activation_prefix}",
        *lora_backward_prefixes("fc1"),
    ]


def dense_targets_attention(target_mode: str) -> bool:
    return target_mode in {"attention_only", "all"}


def dense_targets_mlp(target_mode: str) -> bool:
    return target_mode in {"mlp_only", "all"}


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
        self.memory_values: dict[str, list[dict[str, int]]] = defaultdict(list)
        self.range_stack: list[str] = []
        self.saved_tensor_tracker: SavedTensorMemoryTracker | None = None
        self.max_stage_peak_bytes = 0

    def flush(self) -> None:
        sync(self.device)

    def clear(self) -> None:
        self.values.clear()
        self.memory_values.clear()
        self.max_stage_peak_bytes = 0
        if self.saved_tensor_tracker is not None:
            self.saved_tensor_tracker.clear()

    def current_range(self) -> str:
        return self.range_stack[-1] if self.range_stack else ""

    def memory_summary(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        stage_order = {"step.forward": 0, "step.backward": 1}
        for key, records in sorted(self.memory_values.items(), key=lambda item: (stage_order.get(item[0], 99), item[0])):
            if not records:
                continue
            count = len(records)
            def avg(field: str) -> float:
                return float(sum(int(record[field]) for record in records)) / float(count)

            rows.append(
                {
                    "name": key,
                    "samples": count,
                    "avg_allocated_start_bytes": avg("allocated_start_bytes"),
                    "avg_allocated_end_bytes": avg("allocated_end_bytes"),
                    "avg_allocated_delta_bytes": avg("allocated_delta_bytes"),
                    "avg_reserved_start_bytes": avg("reserved_start_bytes"),
                    "avg_reserved_end_bytes": avg("reserved_end_bytes"),
                    "avg_reserved_delta_bytes": avg("reserved_delta_bytes"),
                    "avg_local_peak_bytes": avg("local_peak_bytes"),
                    "avg_local_peak_delta_bytes": avg("local_peak_delta_bytes"),
                    "max_local_peak_bytes": max(int(record["local_peak_bytes"]) for record in records),
                    "max_global_peak_after_bytes": max(int(record["global_peak_after_bytes"]) for record in records),
                }
            )
        return {"rows": rows, "max_stage_peak_bytes": int(self.max_stage_peak_bytes)}

    @contextmanager
    def time(self, key: str) -> Iterator[None]:
        should_sync = self.timing_mode == "debug_sync" or key.startswith("step.")
        if should_sync:
            sync(self.device)
        start = time.perf_counter()
        pushed = False
        memory_record: dict[str, int] | None = None
        self.range_stack.append(key)
        if self.device.type == "cuda":
            if key in {"step.forward", "step.backward"}:
                memory_record = {
                    "allocated_start_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_start_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "global_peak_start_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                }
                torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.nvtx.range_push(key)
            pushed = True
        try:
            yield
        finally:
            if should_sync:
                sync(self.device)
            if memory_record is not None:
                allocated_end = int(torch.cuda.memory_allocated(self.device))
                reserved_end = int(torch.cuda.memory_reserved(self.device))
                local_peak = int(torch.cuda.max_memory_allocated(self.device))
                self.max_stage_peak_bytes = max(self.max_stage_peak_bytes, local_peak)
                memory_record.update(
                    {
                        "allocated_end_bytes": allocated_end,
                        "allocated_delta_bytes": allocated_end - int(memory_record["allocated_start_bytes"]),
                        "reserved_end_bytes": reserved_end,
                        "reserved_delta_bytes": reserved_end - int(memory_record["reserved_start_bytes"]),
                        "local_peak_bytes": local_peak,
                        "local_peak_delta_bytes": local_peak - int(memory_record["allocated_start_bytes"]),
                        "global_peak_after_bytes": max(int(memory_record["global_peak_start_bytes"]), local_peak),
                    }
                )
                self.memory_values[key].append(memory_record)
            self.values[key] += time.perf_counter() - start
            if pushed:
                torch.cuda.nvtx.range_pop()
            self.range_stack.pop()


def _tensor_storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return int(tensor.data_ptr())


def _tensor_unique_key(tensor: torch.Tensor) -> tuple[str, int, int, int, str]:
    return (
        str(tensor.device),
        int(tensor.data_ptr()),
        int(tensor.numel()),
        int(tensor.element_size()),
        str(tensor.dtype),
    )


def _persistent_storage_ptrs(model: torch.nn.Module) -> set[int]:
    ptrs: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda" and tensor.numel() > 0:
            ptrs.add(_tensor_storage_ptr(tensor))
    return ptrs


def _compact_profile_owner(owner: str) -> str:
    text = owner or "unattributed"
    compact = text.lower()
    compact = re.sub(r"^(forward|backward)\.", "", compact)
    compact = re.sub(r"\blayers\.\d+\.", "", compact)
    return compact


def _projection_bucket(compact: str) -> str:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if name in compact:
            return name
    if "gate_up_lora" in compact:
        return "gate_up"
    for name in ("gate_base", "up_base", "down_base", "gate_lora", "up_lora", "down_lora"):
        if name in compact:
            return name.split("_", 1)[0]
    for name in ("fc1", "fc2", "matrix"):
        if name in compact:
            return name
    return ""


def _operation_bucket(compact: str) -> str:
    if "attention" in compact and (
        "base_asymgemm" in compact or "base frozen asymgemm" in compact or "base_frozen_asymgemm" in compact
    ):
        return "base_torch"
    if "base_asymgemm" in compact or "grouped_base_frozen_asymgemm" in compact or "base_frozen_asymgemm" in compact:
        return "base_asymgemm"
    if "grouped_base_dx_asymgemm" in compact or "base_dx_asymgemm" in compact:
        return "base_asymgemm"
    if "base_torch" in compact or "grouped_base_torch" in compact:
        return "base_torch"
    if "q_proj_base" in compact or "k_proj_base" in compact or "v_proj_base" in compact or "o_proj_base" in compact:
        return "base_torch"
    if "lora_a" in compact:
        return "lora_A"
    if "lora_b" in compact:
        return "lora_B"
    if "gate_up_lora" in compact or "gate_lora" in compact or "up_lora" in compact or "down_lora" in compact:
        return "lora"
    if "add_cast_scale" in compact:
        return "add_cast_scale"
    if "base_lora_add" in compact:
        return "base_lora_add"
    if "activation_silu_mul" in compact or "silu_mul_activation" in compact:
        return "silu_mul_activation"
    if "activation_relu" in compact or "relu" in compact:
        return "relu_activation"
    for name in (
        "scores_matmul",
        "value_matmul",
        "causal_mask",
        "softmax",
        "layernorm",
        "residual_add",
        "route_metadata",
        "pack_tokens",
        "scatter_combine",
        "combine_shared_routed",
        "forward_sft",
        "kt_lora_update",
        "mse",
        "cross_entropy",
    ):
        if name in compact:
            return name
    if "final_norm" in compact:
        return "final_norm"
    if "lm_head" in compact:
        return "lm_head"
    if "embedding" in compact:
        return "embeddings"
    if "router" in compact:
        return "router"
    if "loss" in compact:
        return "loss"
    return ""


def _saved_tensor_bucket(owner: str) -> str:
    text = owner or "unattributed"
    compact = _compact_profile_owner(text)
    projection = _projection_bucket(compact)
    operation = _operation_bucket(compact)

    if "routed_expert" in compact or "shared_expert" in compact:
        expert = "routed_expert" if "routed_expert" in compact else "shared_expert"
        parts = [expert]
        if projection:
            parts.append(projection)
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if "attention" in compact:
        parts = ["attention"]
        if projection:
            parts.append(projection)
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if ".mlp." in compact or compact.startswith("mlp."):
        parts = ["mlp"]
        if projection:
            parts.append(projection)
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if projection:
        parts = [projection]
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if operation:
        return operation
    return text


class SavedTensorMemoryTracker:
    """Memory-only saved-tensor attribution; do not use its run for timing claims."""

    def __init__(self, model: torch.nn.Module, book: StageBook) -> None:
        self.book = book
        self.persistent_storage_ptrs = _persistent_storage_ptrs(model)
        self.unique_seen: dict[tuple[str, int, int, int, str], str] = {}
        self.unique_bytes_by_owner: dict[str, int] = defaultdict(int)
        self.reference_bytes_by_owner: dict[str, int] = defaultdict(int)
        self.save_count_by_owner: dict[str, int] = defaultdict(int)
        self.unique_count_by_owner: dict[str, int] = defaultdict(int)
        self.skipped_persistent_bytes = 0
        self.skipped_non_cuda_bytes = 0

    def clear(self) -> None:
        self.unique_seen.clear()
        self.unique_bytes_by_owner.clear()
        self.reference_bytes_by_owner.clear()
        self.save_count_by_owner.clear()
        self.unique_count_by_owner.clear()
        self.skipped_persistent_bytes = 0
        self.skipped_non_cuda_bytes = 0

    def owner(self) -> str:
        return current_profile_range() or self.book.current_range() or "unattributed"

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        tensor_bytes = nbytes(tensor)
        if tensor_bytes <= 0:
            return tensor
        if tensor.device.type != "cuda":
            self.skipped_non_cuda_bytes += tensor_bytes
            return tensor
        if _tensor_storage_ptr(tensor) in self.persistent_storage_ptrs:
            self.skipped_persistent_bytes += tensor_bytes
            return tensor

        owner = self.owner()
        bucket = _saved_tensor_bucket(owner)
        key = _tensor_unique_key(tensor)
        self.reference_bytes_by_owner[bucket] += tensor_bytes
        self.save_count_by_owner[bucket] += 1
        if key not in self.unique_seen:
            self.unique_seen[key] = bucket
            self.unique_bytes_by_owner[bucket] += tensor_bytes
            self.unique_count_by_owner[bucket] += 1
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def summary(self) -> dict[str, Any]:
        buckets = sorted(
            set(self.reference_bytes_by_owner) | set(self.unique_bytes_by_owner),
            key=lambda name: self.unique_bytes_by_owner.get(name, 0),
            reverse=True,
        )
        rows = [
            {
                "owner": bucket,
                "unique_bytes": int(self.unique_bytes_by_owner.get(bucket, 0)),
                "reference_bytes": int(self.reference_bytes_by_owner.get(bucket, 0)),
                "save_count": int(self.save_count_by_owner.get(bucket, 0)),
                "unique_tensor_count": int(self.unique_count_by_owner.get(bucket, 0)),
            }
            for bucket in buckets
        ]
        return {
            "enabled": True,
            "rows": rows,
            "total_unique_bytes": int(sum(row["unique_bytes"] for row in rows)),
            "total_reference_bytes": int(sum(row["reference_bytes"] for row in rows)),
            "skipped_persistent_bytes": int(self.skipped_persistent_bytes),
            "skipped_non_cuda_bytes": int(self.skipped_non_cuda_bytes),
            "notes": [
                "Saved activation attribution is collected with torch.autograd.graph.saved_tensors_hooks in a memory-only source pass.",
                "unique_bytes deduplicates repeated saves of the same CUDA tensor; reference_bytes shows repeated save references and can overcount live memory.",
                "Persistent model parameters and buffers are excluded from saved activation rows.",
            ],
        }


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


class ProfileTorchFrozenLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    @property
    def pinned_cpu_bytes(self) -> int:
        return 0

    @property
    def weight_hbm_saved_bytes(self) -> int:
        return 0

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return 0

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return nbytes(self.weight)

    def forward(self, x: torch.Tensor, *, backward_prefix: str | None = None, book: StageBook | None = None) -> torch.Tensor:
        if backward_prefix is not None and book is not None:
            return profiled_linear(x, self.weight, None, backward_prefix, book)
        return F.linear(x, self.weight)


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


class _ProfiledCausalMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor, mask: torch.Tensor, fill_value: float, prefix: str, book: StageBook) -> torch.Tensor:
        ctx.save_for_backward(mask)
        ctx.prefix = prefix
        ctx.book = book
        return scores.masked_fill(mask, fill_value)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        (mask,) = ctx.saved_tensors
        with ctx.book.time(f"backward.{ctx.prefix}.grad"):
            grad_scores = grad_output.masked_fill(mask, 0)
        return grad_scores, None, None, None, None


def profiled_causal_mask(scores: torch.Tensor, mask: torch.Tensor, fill_value: float, prefix: str, book: StageBook) -> torch.Tensor:
    return _ProfiledCausalMask.apply(scores, mask, fill_value, prefix, book)


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


class _AutogradNodeRange:
    def __init__(self, name: str, book: StageBook) -> None:
        self.name = name
        self.book = book
        self.record: Any = None
        self.pushed = False

    def __enter__(self) -> "_AutogradNodeRange":
        if self.book.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)
            self.pushed = True
        else:
            self.record = torch.autograd.profiler.record_function(self.name)
            self.record.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.pushed:
            torch.cuda.nvtx.range_pop()
            self.pushed = False
        elif self.record is not None:
            self.record.__exit__(exc_type, exc, tb)
            self.record = None


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _attach_backward_nvtx_ranges(
    outputs: Any,
    name: str,
    book: StageBook,
    *,
    stop_tensors: tuple[torch.Tensor, ...] = (),
) -> None:
    stop_fn_ids = {id(tensor.grad_fn) for tensor in stop_tensors if isinstance(tensor, torch.Tensor) and tensor.grad_fn is not None}
    seen: set[int] = set()
    stack = [tensor.grad_fn for tensor in _iter_tensors(outputs) if tensor.requires_grad and tensor.grad_fn is not None]

    while stack:
        node = stack.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen or node_id in stop_fn_ids:
            continue
        seen.add(node_id)
        if type(node).__name__ != "AccumulateGrad":
            state: dict[str, _AutogradNodeRange] = {}

            def prehook(grad_outputs: Any, *, state: dict[str, _AutogradNodeRange] = state) -> None:
                active = _AutogradNodeRange(name, book)
                active.__enter__()
                state["active"] = active
                return None

            def posthook(grad_inputs: Any, grad_outputs: Any, *, state: dict[str, _AutogradNodeRange] = state) -> None:
                active = state.pop("active", None)
                if active is not None:
                    active.__exit__(None, None, None)
                return None

            try:
                node.register_prehook(prehook)
                node.register_hook(posthook)
            except (AttributeError, RuntimeError):
                pass
        for next_node, _ in getattr(node, "next_functions", ()):
            if next_node is not None:
                stack.append(next_node)


def _backward_label_from_current_range(fallback: str) -> str:
    current = current_profile_range()
    if current.startswith("forward."):
        suffix = current[len("forward.") :]
        return f"backward.{suffix}.grad"
    if current.startswith("backward."):
        return f"{current}.grad"
    return fallback


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
    pinned_w = 0
    pinned = 0
    seen_host_weights: set[int] = set()
    for module in model.modules():
        if hasattr(module, "kt_cpu_weight_bytes"):
            host_w += int(getattr(module, "kt_cpu_weight_bytes", 0))
            pinned += int(getattr(module, "pinned_cpu_bytes", 0))
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
        if isinstance(weight, torch.Tensor):
            weight_bytes = nbytes(weight)
            host_w += weight_bytes
            if weight.is_pinned():
                pinned_w += weight_bytes
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
            "pinned_w_bytes": pinned_w,
            "pinned_total_bytes": pinned,
        },
    }


def tensor_tree_nbytes(value: Any, *, device_type: str | None = None) -> int:
    if isinstance(value, torch.Tensor):
        if device_type is not None and value.device.type != device_type:
            return 0
        return nbytes(value)
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item, device_type=device_type) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(tensor_tree_nbytes(item, device_type=device_type) for item in value)
    return 0


def memory_attribution_report(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tracker: SavedTensorMemoryTracker | None,
) -> dict[str, Any]:
    mem = memory_report(model, device)
    lora_trainable = 0
    other_trainable = 0
    frozen_cuda_params = 0
    gradient_bytes = 0
    for name, param in model.named_parameters():
        param_bytes = nbytes(param) if param.device.type == "cuda" else 0
        if param.requires_grad:
            if "lora" in name.lower():
                lora_trainable += param_bytes
            else:
                other_trainable += param_bytes
            if isinstance(param.grad, torch.Tensor) and param.grad.device.type == "cuda":
                gradient_bytes += nbytes(param.grad)
        else:
            frozen_cuda_params += param_bytes

    frozen_cuda_buffers = int(mem["gpu"]["buffer_bytes"])
    optimizer_state_bytes = tensor_tree_nbytes(optimizer.state, device_type="cuda")
    saved = tracker.summary() if tracker is not None else {"enabled": False, "rows": [], "total_unique_bytes": 0, "total_reference_bytes": 0}
    saved_unique = int(saved.get("total_unique_bytes", 0))
    peak = int(mem["gpu"]["peak_hbm_bytes"])
    known_hbm = frozen_cuda_params + frozen_cuda_buffers + lora_trainable + other_trainable + gradient_bytes + optimizer_state_bytes + saved_unique
    residual = max(0, peak - known_hbm)

    category_rows = [
        {
            "category": "frozen base weights on CPU pinned",
            "bytes": int(mem["cpu"]["pinned_total_bytes"]),
            "memory_space": "CPU pinned",
            "accuracy": "exact",
        },
        {
            "category": "frozen base weights on GPU buffers",
            "bytes": int(frozen_cuda_params + frozen_cuda_buffers),
            "memory_space": "GPU HBM",
            "accuracy": "exact",
        },
        {
            "category": "LoRA trainable params",
            "bytes": int(lora_trainable),
            "memory_space": "GPU HBM",
            "accuracy": "exact",
        },
    ]
    if other_trainable:
        category_rows.append(
            {
                "category": "other trainable params",
                "bytes": int(other_trainable),
                "memory_space": "GPU HBM",
                "accuracy": "exact",
            }
        )
    category_rows.extend(
        [
            {
                "category": "gradients",
                "bytes": int(gradient_bytes),
                "memory_space": "GPU HBM",
                "accuracy": "exact after backward",
            },
            {
                "category": "AdamW optimizer state",
                "bytes": int(optimizer_state_bytes),
                "memory_space": "GPU HBM",
                "accuracy": "exact after optimizer step",
            },
            {
                "category": "saved forward activations by semantic op",
                "bytes": int(saved_unique),
                "memory_space": "GPU HBM",
                "accuracy": "hook-attributed unique saved tensors" if tracker is not None else "not collected",
            },
            {
                "category": "allocator peak / unattributed",
                "bytes": int(residual),
                "memory_space": "GPU HBM",
                "accuracy": "estimated residual",
            },
        ]
    )
    return {
        "enabled": tracker is not None,
        "categories": {
            "rows": category_rows,
            "known_hbm_bytes": int(known_hbm),
            "peak_hbm_bytes": peak,
            "unattributed_hbm_bytes": int(residual),
        },
        "saved_activations": saved,
        "notes": [
            "Model, gradient, and optimizer rows are tensor-size accounting.",
            "Saved activation rows require --memory-attribution and are memory-only; do not use that pass for timing claims.",
            "allocator peak / unattributed is a residual against torch.cuda.max_memory_allocated and includes temporaries, workspaces, allocator effects, inputs/targets, and phase overlap.",
        ],
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
    counter: dict[str, int] = defaultdict(int)

    def timed_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        idx = counter["linear"]
        counter["linear"] += 1
        prefix = "fc1" if idx % 2 == 0 else "fc2"
        return profiled_lora_linear(self, x, prefix, book)

    originals.append((mlp.AsymLoRALinear, "forward", mlp.AsymLoRALinear.forward))
    mlp.AsymLoRALinear.forward = timed_forward
    originals.append((mlp.TorchLoRALinear, "forward", mlp.TorchLoRALinear.forward))
    mlp.TorchLoRALinear.forward = timed_forward

    def timed_mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        return _profiled_mlp_lora_forward(self, x, book)

    originals.append((mlp.AsymMLP, "forward", mlp.AsymMLP.forward))
    mlp.AsymMLP.forward = timed_mlp_forward
    originals.append((mlp.TorchMLP, "forward", mlp.TorchMLP.forward))
    mlp.TorchMLP.forward = timed_mlp_forward
    return originals


def restore(originals: list[tuple[Any, str, Any]]) -> None:
    for obj, attr, original in reversed(originals):
        setattr(obj, attr, original)


def _lora_adapter_tensors(module: Any, prefix: str) -> LoRAAdapterTensors:
    if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
        return LoRAAdapterTensors(module.lora_a, module.lora_b, float(module.scaling))
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        return LoRAAdapterTensors(module.lora_A["default"].weight, module.lora_B["default"].weight, float(module.scaling))
    raise TypeError(f"unsupported LoRA adapter layout for prefix={prefix!r}: {type(module).__name__}")


def _profiled_lora_base(module: Any, x: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    if hasattr(module, "base"):
        with book.time(f"forward.{prefix}.base_frozen_asymgemm"):
            setattr(module.base, "profile_name", prefix)
            return module.base(x)
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "profile_name"):
        with book.time(f"forward.{prefix}.base_frozen_asymgemm"):
            module.base_layer.profile_name = prefix
            return module.base_layer(x)
    if hasattr(module, "base_weight"):
        with book.time(f"forward.{prefix}.base_torch"):
            return profiled_linear(x, module.base_weight, None, f"{prefix}.base_torch", book)
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        with book.time(f"forward.{prefix}.base_torch"):
            return profiled_linear(x, module.base_layer.weight, None, f"{prefix}.base_torch", book)
    raise TypeError(f"unsupported LoRA base layout for prefix={prefix!r}: {type(module).__name__}")


def profiled_lora_linear(module: Any, x: torch.Tensor, prefix: str, book: StageBook) -> torch.Tensor:
    base = _profiled_lora_base(module, x, prefix, book)
    adapter = _lora_adapter_tensors(module, prefix)
    lora_input = x.to(dtype=adapter.a.dtype)
    with book.time(f"forward.{prefix}.lora_A"):
        low_rank = profiled_linear(lora_input, adapter.a, None, f"{prefix}.lora_A", book)
    with book.time(f"forward.{prefix}.lora_B"):
        lora_raw = profiled_linear(low_rank, adapter.b, None, f"{prefix}.lora_B", book)
    with book.time(f"forward.{prefix}.add_cast_scale"):
        lora = profiled_scale_cast(lora_raw, adapter.scaling, base.dtype, f"{prefix}.add_cast_scale", book)
        return profiled_residual_add(base, lora, f"{prefix}.base_lora_add", book)


def _profiled_mlp_lora_forward(model: Any, x: torch.Tensor, book: StageBook) -> torch.Tensor:
    hidden = profiled_lora_linear(model.fc1, x, "fc1", book)
    with book.time("forward.activation_relu"):
        hidden = profiled_relu(hidden, "activation_relu", book)
    return profiled_lora_linear(model.fc2, hidden, "fc2", book)


def make_matrix_lora_forward_fn(book: StageBook) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]:
    def forward_fn(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return profiled_lora_linear(model, x, "matrix", book)

    return forward_fn


def make_mlp_lora_forward_fn(book: StageBook) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]:
    def forward_fn(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return _profiled_mlp_lora_forward(model, x, book)

    return forward_fn


def profile_mlp(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.mlp import AsymMLP, TorchMLP, lora_parameters

    clear(device)
    stats = AsymExecutionStats()
    default_tokens, in_features, hidden, out_features, rank = (64, 128, 256, 128, 8) if device.type == "cuda" else (4, 16, 32, 16, 4)
    tokens = requested_tokens(args, default_tokens)
    torch.manual_seed(0)
    w1 = torch.randn(hidden, in_features, dtype=dtype)
    w2 = torch.randn(out_features, hidden, dtype=dtype)
    lora_dtype = profile_lora_dtype(args)
    if args.backend == "torch":
        model = TorchMLP(w1, w2, rank=rank, alpha=16.0, device=device, dtype=dtype, lora_dtype=lora_dtype)
    else:
        model = AsymMLP(
            w1,
            w2,
            rank=rank,
            alpha=16.0,
            backend=args.backend,
            stats=stats,
            device=device,
            dtype=dtype,
            lora_dtype=lora_dtype,
            precision=args.precision,
        )
    optimizer = make_lora_optimizer(model, lora_parameters)

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
        forward_keys=mlp_lora_forward_keys(),
        backward_group_prefixes=mlp_lora_backward_prefixes(),
        config={
            "tokens": tokens,
            "in_features": in_features,
            "hidden": hidden,
            "out_features": out_features,
            "rank": rank,
            "lora_dtype": str(lora_dtype),
        },
        stage_memory=book.memory_summary(),
        memory_attribution=memory_attribution_report(model, optimizer, device, book.saved_tensor_tracker),
    )


def _lora_sft_config(raw: dict[str, Any], *, rank: int, alpha: float) -> dict[str, Any]:
    config = dict(raw)
    if "hidden_features" in config:
        base_params = config["hidden_features"] * (config["in_features"] + config["out_features"])
        lora_params = rank * (config["in_features"] + config["hidden_features"])
        lora_params += rank * (config["hidden_features"] + config["out_features"])
    else:
        base_params = config["in_features"] * config["out_features"]
        lora_params = rank * (config["in_features"] + config["out_features"])
    config.update(
        {
            "base_parameter_count": int(base_params),
            "base_parameter_billions": float(base_params) / 1_000_000_000.0,
            "lora_rank": int(rank),
            "lora_alpha": float(alpha),
            "trainable_lora_elements": int(lora_params),
            "total_model_elements": int(base_params + lora_params),
            "profile_goal": "lora_sft",
        }
    )
    return config


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _shape_args(args: argparse.Namespace) -> tuple[int, int, int]:
    return (
        int(getattr(args, "real_tokens", 0) or 0),
        int(getattr(args, "real_batch_size", 0) or 0),
        int(getattr(args, "real_seq_len", 0) or 0),
    )


def _resolve_regression_shape(args: argparse.Namespace, config: dict[str, Any]) -> None:
    explicit_tokens, batch_size, seq_len = _shape_args(args)
    if explicit_tokens > 0:
        config["tokens"] = _positive_int(explicit_tokens, "tokens")
        if batch_size > 0 and seq_len > 0 and batch_size * seq_len == explicit_tokens:
            config["batch_size"] = batch_size
            config["seq_len"] = seq_len
        else:
            config.pop("batch_size", None)
            config.pop("seq_len", None)
        return

    if "batch_size" in config or "seq_len" in config:
        batch = _positive_int(batch_size or int(config.get("batch_size", 0)), "batch_size")
        seq = _positive_int(seq_len or int(config.get("seq_len", 0)), "seq_len")
        config["batch_size"] = batch
        config["seq_len"] = seq
        config["tokens"] = batch * seq
    else:
        config["tokens"] = _positive_int(int(config["tokens"]), "tokens")


def _synthetic_lora_config(args: argparse.Namespace, raw: dict[str, Any], workload: str) -> dict[str, Any]:
    config = dict(raw)
    _resolve_regression_shape(args, config)

    hidden_dim = int(getattr(args, "real_hidden_dim", 0) or 0)
    if hidden_dim > 0 and workload in {"mm_3b", "mlp_3b"}:
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        config["hidden_dim"] = hidden_dim
        config["in_features"] = hidden_dim
        config["out_features"] = hidden_dim
        if "hidden_features" in config:
            intermediate_dim = int(getattr(args, "real_mlp_intermediate_dim", 0) or 0)
            if intermediate_dim <= 0:
                expansion = int(getattr(args, "real_mlp_expansion", DEFAULT_LORA_MLP_EXPANSION) or DEFAULT_LORA_MLP_EXPANSION)
                expansion = _positive_int(expansion, "mlp_expansion")
                intermediate_dim = hidden_dim * expansion
            intermediate_dim = _positive_int(intermediate_dim, "mlp_intermediate_dim")
            config["hidden_features"] = intermediate_dim
            config["mlp_expansion"] = float(intermediate_dim) / float(hidden_dim)
    return config


def lora_hparams(args: argparse.Namespace) -> tuple[int, float]:
    rank = _positive_int(int(args.real_lora_rank), "lora_rank")
    alpha = float(args.real_lora_alpha)
    if alpha <= 0.0:
        raise ValueError(f"lora_alpha must be > 0, got {alpha}")
    return rank, alpha


def parse_target_modules(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    modules = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    return modules or None


def _split_module_selector(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return tuple(dict.fromkeys(part.strip().replace("-", "_") for part in normalized.split(",") if part.strip()))


def dense_selector_names(selector: str | None, *, default: str, purpose: str) -> set[str]:
    selectors = _split_module_selector(selector) or (default,)
    names: set[str] = set()
    for raw in selectors:
        key = raw.lower().replace("-", "_")
        if key in {"all", "default"}:
            names.update(DENSE_ALL_PROJECTIONS)
        elif key in {"attention", "attention_only"}:
            names.update(DENSE_ATTENTION_PROJECTIONS)
        elif key in {"mlp", "mlp_only"}:
            names.update(DENSE_MLP_PROJECTIONS)
        elif key == "none":
            continue
        else:
            names.add(raw.rsplit(".", 1)[-1].replace("-", "_"))
    invalid = sorted(names - set(DENSE_ALL_PROJECTIONS))
    if invalid:
        raise ValueError(f"unsupported dense {purpose} module suffixes {invalid}; expected entries from {DENSE_ALL_PROJECTIONS}")
    return names


def moe_selector_groups(selector: str | None, *, default: str, purpose: str) -> set[str]:
    selectors = _split_module_selector(selector) or (default,)
    groups: set[str] = set()
    for raw in selectors:
        key = raw.lower().replace("-", "_")
        if key in {"all", "mlp", "experts", "expert_mlp", "default"}:
            groups.update(("routed_experts", "shared_experts"))
        elif key in {"routed", "routed_expert", "routed_experts"}:
            groups.add("routed_experts")
        elif key in {"shared", "shared_expert", "shared_experts"}:
            groups.add("shared_experts")
        elif key == "none":
            continue
        else:
            raise ValueError(
                f"unsupported MoE {purpose} selector {raw!r}; expected all, mlp, routed_experts, shared_experts, or none"
            )
    return groups


def profile_lora_dtype(args: argparse.Namespace) -> torch.dtype:
    from asym_gemm.training.lora import normalize_lora_dtype

    return normalize_lora_dtype(getattr(args, "lora_dtype", "bf16"))


def lora_config_with_dtype(args: argparse.Namespace, raw: dict[str, Any], *, rank: int, alpha: float) -> dict[str, Any]:
    config = _lora_sft_config(raw, rank=rank, alpha=alpha)
    config["lora_dtype"] = str(profile_lora_dtype(args))
    return config


def make_lora_optimizer(
    model: torch.nn.Module,
    lora_parameters_fn: Callable[[torch.nn.Module], list[torch.nn.Parameter]],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(lora_parameters_fn(model), lr=1e-2)


class LoraMSEProfileRunner:
    def __init__(self, args: argparse.Namespace, device: torch.device, dtype: torch.dtype, stats: Any) -> None:
        self.args = args
        self.device = device
        self.dtype = dtype
        self.stats = stats

    def make_batch_fn(self, shape: RegressionBatchShape) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(*shape.input_shape(), device=self.device, dtype=self.dtype, requires_grad=True)
            target = torch.randn(*shape.target_shape(), device=self.device, dtype=self.dtype)
            return x, target

        return make_batch

    def loss_fn(self, book: StageBook) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return profiled_mse_loss(prediction.float(), target.float(), "loss.mse", book)

        return loss

    def profile(
        self,
        *,
        workload: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        shape: RegressionBatchShape,
        forward_fn_factory: Callable[[StageBook], Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]],
        forward_keys: list[str],
        backward_group_prefixes: list[str],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        book = StageBook(self.device, timing_mode=self.args.timing_mode)
        forward_fn = forward_fn_factory(book)
        run_profile_steps(
            book,
            model,
            optimizer,
            self.make_batch_fn(shape),
            forward_fn,
            self.loss_fn(book),
            self.args,
            reset_stats=lambda: reset_execution_stats(self.stats),
        )
        avg = average(book.values, self.args.measure_steps)
        return build_report(
            workload,
            model,
            self.device,
            avg,
            self.stats.as_dict(),
            forward_keys=forward_keys,
            backward_group_prefixes=backward_group_prefixes,
            config=config,
            stage_memory=book.memory_summary(),
            memory_attribution=memory_attribution_report(model, optimizer, self.device, book.saved_tensor_tracker),
        )


def profile_mm_lora(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, workload: str) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.mlp import AsymLoRALinear, TorchLoRALinear, lora_parameters

    clear(device)
    stats = AsymExecutionStats()
    cfg = _synthetic_lora_config(args, MM_CONFIGS[workload], workload)
    rank, alpha = lora_hparams(args)
    torch.manual_seed(101)
    weight = torch.randn(cfg["out_features"], cfg["in_features"], dtype=dtype)
    lora_dtype = profile_lora_dtype(args)
    if args.backend == "torch":
        model = TorchLoRALinear(weight, rank=rank, alpha=alpha, device=device, dtype=dtype, lora_dtype=lora_dtype)
    else:
        model = AsymLoRALinear(
            weight,
            rank=rank,
            alpha=alpha,
            backend=args.backend,
            stats=stats,
            device=device,
            dtype=dtype,
            lora_dtype=lora_dtype,
            precision=args.precision,
        )
    del weight
    optimizer = make_lora_optimizer(model, lora_parameters)
    return LoraMSEProfileRunner(args, device, dtype, stats).profile(
        workload=workload,
        model=model,
        optimizer=optimizer,
        shape=RegressionBatchShape.from_config(cfg),
        forward_fn_factory=make_matrix_lora_forward_fn,
        forward_keys=lora_forward_keys("matrix"),
        backward_group_prefixes=["backward.loss.mse", *lora_backward_prefixes("matrix")],
        config=lora_config_with_dtype(args, cfg, rank=rank, alpha=alpha),
    )


def profile_mlp_lora(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, workload: str) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.mlp import AsymMLP, TorchMLP, lora_parameters

    clear(device)
    stats = AsymExecutionStats()
    cfg = _synthetic_lora_config(args, MLP_CONFIGS[workload], workload)
    rank, alpha = lora_hparams(args)
    torch.manual_seed(202)
    w1 = torch.randn(cfg["hidden_features"], cfg["in_features"], dtype=dtype)
    w2 = torch.randn(cfg["out_features"], cfg["hidden_features"], dtype=dtype)
    lora_dtype = profile_lora_dtype(args)
    if args.backend == "torch":
        model = TorchMLP(w1, w2, rank=rank, alpha=alpha, device=device, dtype=dtype, lora_dtype=lora_dtype)
    else:
        model = AsymMLP(
            w1,
            w2,
            rank=rank,
            alpha=alpha,
            backend=args.backend,
            stats=stats,
            device=device,
            dtype=dtype,
            lora_dtype=lora_dtype,
            precision=args.precision,
        )
    del w1, w2
    optimizer = make_lora_optimizer(model, lora_parameters)

    return LoraMSEProfileRunner(args, device, dtype, stats).profile(
        workload=workload,
        model=model,
        optimizer=optimizer,
        shape=RegressionBatchShape.from_config(cfg),
        forward_fn_factory=make_mlp_lora_forward_fn,
        forward_keys=mlp_lora_forward_keys(),
        backward_group_prefixes=mlp_lora_backward_prefixes(),
        config=lora_config_with_dtype(args, cfg, rank=rank, alpha=alpha),
    )


def profile_mlp_1b(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return profile_mlp_lora(args, device, dtype, "mlp_1b")


def set_profile_names(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        setattr(module, "_m4_profile_name", name)


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
        return profiled_lora_linear(self, x, linear_prefix(self).removeprefix("forward."), book)

    originals.append((dense.AsymLoRALinear, "forward", original_linear))
    dense.AsymLoRALinear.forward = timed_linear
    originals.append((dense.TorchLoRALinear, "forward", dense.TorchLoRALinear.forward))
    dense.TorchLoRALinear.forward = timed_linear

    def profiled_projection(module: Any, x: torch.Tensor, prefix: str) -> torch.Tensor:
        if hasattr(module, "lora_A") or hasattr(module, "lora_a"):
            return module(x)
        if hasattr(module, "host_weight") and hasattr(module, "profile_name"):
            module.profile_name = prefix
            with book.time(f"forward.{prefix}.base_frozen_asymgemm"):
                return module(x)
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            with book.time(f"forward.{prefix}.base_torch"):
                return profiled_linear(x, weight, None, f"{prefix}.base_torch", book)
        raise TypeError(f"unsupported dense projection for prefix={prefix!r}: {type(module).__name__}")

    original_attn = dense.TinySelfAttention.forward

    def timed_attn(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        name = str(getattr(self, "_m4_profile_name", "attention"))
        layer = _layer_prefix_from_module_name(name)
        prefix = f"forward.{layer}.attention" if layer else "forward.attention"
        bprefix = prefix.removeprefix("forward.")
        batch, seq, hidden = hidden_states.shape
        q = profiled_projection(self.q_proj, hidden_states, f"{bprefix}.q_proj").view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = profiled_projection(self.k_proj, hidden_states, f"{bprefix}.k_proj").view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = profiled_projection(self.v_proj, hidden_states, f"{bprefix}.v_proj").view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        with book.time(f"{prefix}.scores_matmul"):
            scores = profiled_matmul(q.float(), k.float().transpose(-2, -1), f"{bprefix}.scores_matmul", book) / (float(self.head_dim) ** 0.5)
        with book.time(f"{prefix}.causal_mask"):
            mask = torch.triu(torch.ones(seq, seq, device=hidden_states.device, dtype=torch.bool), diagonal=1)
            scores = profiled_causal_mask(scores, mask, torch.finfo(scores.dtype).min, f"{bprefix}.causal_mask", book)
        with book.time(f"{prefix}.softmax"):
            probs = profiled_softmax(scores, -1, f"{bprefix}.softmax", book)
        with book.time(f"{prefix}.value_matmul"):
            context = profiled_matmul(probs, v.float(), f"{bprefix}.value_matmul", book).transpose(1, 2).contiguous().view(batch, seq, hidden)
        return profiled_projection(self.o_proj, context.to(dtype=hidden_states.dtype), f"{bprefix}.o_proj")

    originals.append((dense.TinySelfAttention, "forward", original_attn))
    dense.TinySelfAttention.forward = timed_attn

    original_mlp = dense.TinyMLP.forward

    def timed_mlp(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        name = str(getattr(self, "_m4_profile_name", "mlp"))
        layer = _layer_prefix_from_module_name(name)
        prefix = f"forward.{layer}.mlp" if layer else "forward.mlp"
        bprefix = prefix.removeprefix("forward.")
        gate = profiled_projection(self.gate_proj, hidden_states, f"{bprefix}.gate_proj")
        up = profiled_projection(self.up_proj, hidden_states, f"{bprefix}.up_proj")
        with book.time(f"{prefix}.silu_mul_activation"):
            activated = profiled_silu_mul(gate, up, f"{bprefix}.silu_mul_activation", book)
        return profiled_projection(self.down_proj, activated.to(dtype=hidden_states.dtype), f"{bprefix}.down_proj")

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
            if bool(getattr(self, "gradient_checkpointing", False)) and hidden_states.requires_grad:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
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


def dense_lora_forward_keys(prefix: str, *, offloaded: bool) -> list[str]:
    base_op = "base_frozen_asymgemm" if offloaded else "base_torch"
    return [
        f"forward.{prefix}.{base_op}",
        f"forward.{prefix}.lora_A",
        f"forward.{prefix}.lora_B",
        f"forward.{prefix}.add_cast_scale",
    ]


def dense_lora_backward_prefixes(prefix: str, *, offloaded: bool) -> list[str]:
    base_op = "base_dx_asymgemm" if offloaded else "base_torch"
    return [
        f"backward.{prefix}.{base_op}",
        f"backward.{prefix}.base_lora_add",
        f"backward.{prefix}.add_cast_scale",
        f"backward.{prefix}.lora_B",
        f"backward.{prefix}.lora_A",
    ]


def dense_base_forward_key(prefix: str, *, offloaded: bool) -> str:
    return f"forward.{prefix}.{'base_frozen_asymgemm' if offloaded else 'base_torch'}"


def dense_base_backward_prefix(prefix: str, *, offloaded: bool) -> str:
    return f"backward.{prefix}.{'base_dx_asymgemm' if offloaded else 'base_torch'}"


def dense_projection_forward_keys(
    layer: str,
    scope: str,
    projections: tuple[str, ...],
    *,
    target_names: set[str],
    offload_names: set[str],
) -> list[str]:
    keys: list[str] = []
    for projection in projections:
        prefix = f"{layer}.{scope}.{projection}"
        offloaded = projection in offload_names
        if projection in target_names:
            keys.extend(dense_lora_forward_keys(prefix, offloaded=offloaded))
        else:
            keys.append(dense_base_forward_key(prefix, offloaded=offloaded))
    return keys


def dense_projection_backward_prefixes(
    layer: str,
    scope: str,
    projections: tuple[str, ...],
    *,
    target_names: set[str],
    offload_names: set[str],
) -> list[str]:
    keys: list[str] = []
    for projection in projections:
        prefix = f"{layer}.{scope}.{projection}"
        offloaded = projection in offload_names
        if projection in target_names:
            keys.extend(dense_lora_backward_prefixes(prefix, offloaded=offloaded))
        else:
            keys.append(dense_base_backward_prefix(prefix, offloaded=offloaded))
    return keys


def dense_forward_keys(num_layers: int, target_names: set[str], offload_names: set[str]) -> list[str]:
    keys = ["forward.embeddings"]
    for layer_idx in range(num_layers):
        layer = f"layers.{layer_idx}"
        keys.append(f"forward.{layer}.attention.layernorm")
        keys.extend(
            dense_projection_forward_keys(
                layer,
                "attention",
                DENSE_ATTENTION_PROJECTIONS[:3],
                target_names=target_names,
                offload_names=offload_names,
            )
        )
        keys.extend(
            [
                f"forward.{layer}.attention.scores_matmul",
                f"forward.{layer}.attention.causal_mask",
                f"forward.{layer}.attention.softmax",
                f"forward.{layer}.attention.value_matmul",
            ]
        )
        keys.extend(
            dense_projection_forward_keys(
                layer,
                "attention",
                DENSE_ATTENTION_PROJECTIONS[3:],
                target_names=target_names,
                offload_names=offload_names,
            )
        )
        keys.append(f"forward.{layer}.attention.residual_add")
        keys.append(f"forward.{layer}.mlp.layernorm")
        keys.extend(
            dense_projection_forward_keys(
                layer,
                "mlp",
                DENSE_MLP_PROJECTIONS[:2],
                target_names=target_names,
                offload_names=offload_names,
            )
        )
        keys.append(f"forward.{layer}.mlp.silu_mul_activation")
        keys.extend(
            dense_projection_forward_keys(
                layer,
                "mlp",
                DENSE_MLP_PROJECTIONS[2:],
                target_names=target_names,
                offload_names=offload_names,
            )
        )
        keys.append(f"forward.{layer}.mlp.residual_add")
    keys.extend(["forward.final_norm", "forward.lm_head"])
    return keys


def dense_backward_prefixes(num_layers: int, target_names: set[str], offload_names: set[str]) -> list[str]:
    keys = ["backward.loss.cross_entropy", "backward.lm_head", "backward.final_norm"]
    for layer_idx in reversed(range(num_layers)):
        layer = f"layers.{layer_idx}"
        keys.extend(
            [
                f"backward.{layer}.mlp.residual_add",
                *dense_projection_backward_prefixes(
                    layer,
                    "mlp",
                    ("down_proj",),
                    target_names=target_names,
                    offload_names=offload_names,
                ),
                f"backward.{layer}.mlp.silu_mul_activation",
                *dense_projection_backward_prefixes(
                    layer,
                    "mlp",
                    ("up_proj", "gate_proj"),
                    target_names=target_names,
                    offload_names=offload_names,
                ),
                f"backward.{layer}.mlp.layernorm",
                f"backward.{layer}.attention.residual_add",
                *dense_projection_backward_prefixes(
                    layer,
                    "attention",
                    ("o_proj",),
                    target_names=target_names,
                    offload_names=offload_names,
                ),
                f"backward.{layer}.attention.value_matmul",
                f"backward.{layer}.attention.softmax",
                f"backward.{layer}.attention.causal_mask",
                f"backward.{layer}.attention.scores_matmul",
                *dense_projection_backward_prefixes(
                    layer,
                    "attention",
                    ("v_proj", "k_proj", "q_proj"),
                    target_names=target_names,
                    offload_names=offload_names,
                ),
                f"backward.{layer}.attention.layernorm",
            ]
        )
    return keys


def profile_dense(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.dense import (
        AsymTinyDenseLLM,
        MICRO_DENSE_LLM_CONFIG,
        TorchTinyDenseLLM,
        make_inputs,
        make_tiny_dense_weights,
    )

    clear(device)
    config = getattr(args, "_dense_config_override", MICRO_DENSE_LLM_CONFIG)
    if not hasattr(args, "_dense_config_override"):
        config = replace(
            config,
            batch_size=int(args.real_batch_size),
            seq_len=int(args.real_seq_len),
            lora_rank=int(args.real_lora_rank),
            lora_alpha=float(args.real_lora_alpha),
        )
    workload_name = str(getattr(args, "_workload_name_override", "m4_2_dense_llm"))
    config_extra = dict(getattr(args, "_config_extra", {}))
    target_mode = str(getattr(args, "target_preset", getattr(args, "dense_target_mode", DEFAULT_DENSE_TARGET_MODE)))
    if target_mode not in DENSE_TARGET_MODES:
        raise ValueError(f"target_preset={target_mode!r} must be one of {DENSE_TARGET_MODES}")
    target_selector = str(getattr(args, "target_modules", DEFAULT_TARGET_MODULES) or DEFAULT_TARGET_MODULES)
    offload_selector = str(getattr(args, "offload_modules", DEFAULT_DENSE_OFFLOAD_MODULES) or DEFAULT_DENSE_OFFLOAD_MODULES)
    target_names = dense_selector_names(target_selector, default=target_mode, purpose="target")
    offload_names = dense_selector_names(offload_selector, default=DEFAULT_DENSE_OFFLOAD_MODULES, purpose="offload")
    lora_dtype = profile_lora_dtype(args)
    config_extra["target_mode"] = target_mode
    config_extra["target_preset"] = target_mode
    config_extra["target_modules"] = target_selector
    config_extra["offload_modules"] = offload_selector
    config_extra["lora_dtype"] = str(lora_dtype)
    config_extra["dense_target_names"] = sorted(target_names)
    config_extra["dense_offload_names"] = sorted(offload_names)
    config_extra["dense_offload_scope"] = offload_selector
    config_extra["activation_recompute"] = bool(getattr(args, "activation_recompute", False))
    stats = AsymExecutionStats()
    weights = make_tiny_dense_weights(config, seed=1, dtype=dtype)
    if args.backend == "torch":
        model = TorchTinyDenseLLM(
            weights,
            config=config,
            target_mode=target_mode,
            target_modules=target_selector,
            device=device,
            dtype=dtype,
            lora_seed=2,
            gradient_checkpointing=bool(getattr(args, "activation_recompute", False)),
            lora_dtype=lora_dtype,
        )
    else:
        model = AsymTinyDenseLLM(
            weights,
            config=config,
            target_mode=target_mode,
            target_modules=target_selector,
            offload_modules=offload_selector,
            backend=args.backend,
            stats=stats,
            device=device,
            dtype=dtype,
            lora_seed=2,
            gradient_checkpointing=bool(getattr(args, "activation_recompute", False)),
            lora_dtype=lora_dtype,
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
        forward_keys=dense_forward_keys(config.num_layers, target_names, offload_names),
        backward_group_prefixes=dense_backward_prefixes(config.num_layers, target_names, offload_names),
        config={**asdict(config), **config_extra},
        stage_memory=book.memory_summary(),
        memory_attribution=memory_attribution_report(model, optimizer, device, book.saved_tensor_tracker),
    )


def patch_moe_forward(book: StageBook) -> list[tuple[Any, str, Any]]:
    import asym_gemm.training.moe as moe

    originals: list[tuple[Any, str, Any]] = []

    original_packed_gate_up = moe.PackedExpertLoRA.forward_gate_up
    original_packed_forward = moe.PackedExpertLoRA.forward

    def timed_packed_gate_up(
        self: Any,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        out_dtype: torch.dtype,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_out, up_out = original_packed_gate_up(self, x, offsets, experts, out_dtype, *args, **kwargs)
        _attach_backward_nvtx_ranges(
            (gate_out, up_out),
            _backward_label_from_current_range("backward.routed_expert.gate_up_lora.grad"),
            book,
            stop_tensors=(x, self.gate_lora_a, self.up_lora_a, self.gate_lora_b, self.up_lora_b),
        )
        return gate_out, up_out

    def timed_packed_forward(
        self: Any,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        prefix: str,
        out_dtype: torch.dtype,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        out = original_packed_forward(self, x, offsets, experts, prefix, out_dtype, *args, **kwargs)
        _attach_backward_nvtx_ranges(
            out,
            _backward_label_from_current_range(f"backward.routed_expert.{prefix}_lora.grad"),
            book,
            stop_tensors=(x, self._weight(prefix, "a"), self._weight(prefix, "b")),
        )
        return out

    originals.append((moe.PackedExpertLoRA, "forward_gate_up", original_packed_gate_up))
    moe.PackedExpertLoRA.forward_gate_up = timed_packed_gate_up
    originals.append((moe.PackedExpertLoRA, "forward", original_packed_forward))
    moe.PackedExpertLoRA.forward = timed_packed_forward

    original_grouped_compact = moe.AsymTinyMoELayer._run_grouped_compact

    def timed_grouped_compact(
        self: Any,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        gate_base: torch.nn.Module,
        up_base: torch.nn.Module,
        down_base: torch.nn.Module,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> torch.Tensor:
        if packed.numel() == 0:
            return packed.new_empty((0, self.config.hidden_size))

        layer_name = str(getattr(self, "_m4_profile_name", ""))
        expert_scope = "shared_expert" if shared else "routed_expert"
        profile_prefix = f"{layer_name}.{expert_scope}" if layer_name else expert_scope
        gate_base.profile_name = f"{profile_prefix}.gate_base"
        up_base.profile_name = f"{profile_prefix}.up_base"
        down_base.profile_name = f"{profile_prefix}.down_base"

        gate = gate_base(packed, offsets, experts, dense_experts=dense_experts)
        up = up_base(packed, offsets, experts, dense_experts=dense_experts)
        range_prefix = f"forward.{profile_prefix}"
        if shared:
            assert self.shared_expert_lora is not None
            lora_metadata = self.shared_expert_lora.prepare_metadata(offsets, experts, dense_experts=dense_experts)
            with moe.prof_range(f"{range_prefix}.gate_up_lora"):
                gate_lora, up_lora = self.shared_expert_lora.forward_gate_up(
                    packed,
                    offsets,
                    experts,
                    packed.dtype,
                    metadata=lora_metadata,
                )
        else:
            lora_metadata = self.expert_lora.prepare_metadata(offsets, experts, dense_experts=dense_experts)
            with moe.prof_range(f"{range_prefix}.gate_up_lora"):
                gate_lora, up_lora = self.expert_lora.forward_gate_up(
                    packed,
                    offsets,
                    experts,
                    packed.dtype,
                    metadata=lora_metadata,
                )
        gate_base_out = gate
        up_base_out = up
        gate = gate + gate_lora
        up = up + up_lora
        _attach_backward_nvtx_ranges(
            gate,
            f"backward.{profile_prefix}.gate_base_lora_add.grad",
            book,
            stop_tensors=(gate_base_out, gate_lora),
        )
        _attach_backward_nvtx_ranges(
            up,
            f"backward.{profile_prefix}.up_base_lora_add.grad",
            book,
            stop_tensors=(up_base_out, up_lora),
        )

        with moe.prof_range(f"{range_prefix}.activation_silu_mul"):
            activated = (F.silu(gate.float()) * up.float()).to(dtype=packed.dtype)
            _attach_backward_nvtx_ranges(
                activated,
                _backward_label_from_current_range(f"backward.{profile_prefix}.activation_silu_mul.grad"),
                book,
                stop_tensors=(gate, up),
            )
        down = down_base(activated.contiguous(), offsets, experts, dense_experts=dense_experts)
        if shared:
            with moe.prof_range(f"{range_prefix}.down_lora"):
                down_lora = self._lora_shared(
                    activated,
                    offsets,
                    "down",
                    packed.dtype,
                    experts=experts,
                    lora_metadata=lora_metadata,
                )
        else:
            with moe.prof_range(f"{range_prefix}.down_lora"):
                down_lora = self.expert_lora(activated, offsets, experts, "down", packed.dtype, metadata=lora_metadata)
        down_base_out = down
        down = down + down_lora
        _attach_backward_nvtx_ranges(
            down,
            f"backward.{profile_prefix}.down_base_lora_add.grad",
            book,
            stop_tensors=(down_base_out, down_lora),
        )
        return down

    originals.append((moe.AsymTinyMoELayer, "_run_grouped_compact", original_grouped_compact))
    moe.AsymTinyMoELayer._run_grouped_compact = timed_grouped_compact

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
            scores = profiled_causal_mask(scores, mask, torch.finfo(scores.dtype).min, "attention.causal_mask", book)
        with book.time("forward.attention.softmax"):
            probs = profiled_softmax(scores, -1, "attention.softmax", book)
        with book.time("forward.attention.value_matmul"):
            context = profiled_matmul(probs, v.float(), "attention.value_matmul", book).transpose(1, 2).contiguous().view(batch, seq, hidden)
        with book.time("forward.attention.o_proj_base"):
            out = profiled_linear(context.to(dtype=hidden_states.dtype), self.o_proj.weight, None, "attention.o_proj_base", book)
        return out.squeeze(0) if original_dim == 2 else out

    originals.append((moe.TinySelfAttention, "forward", original_attn))
    moe.TinySelfAttention.forward = timed_attn

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

    for layer_cls in (moe.AsymTinyMoELayer, moe.TorchTinyMoELayer, moe.KTTinyMoELayer):
        originals.append((layer_cls, "forward", layer_cls.forward))
        layer_cls.forward = timed_layer_forward

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
            with book.time("forward.routed_expert.grouped"):
                expert_output = self._run_contiguous(packed, metadata)
            with book.time("forward.scatter_combine"):
                routed_out = profiled_scatter_tokens(expert_output, metadata, "contiguous", book)
        else:
            with book.time("forward.pack_tokens"):
                packed = profiled_pack_tokens(flat, metadata, "masked", book)
            with book.time("forward.routed_expert.grouped"):
                expert_output = self._run_masked(packed, metadata)
            with book.time("forward.scatter_combine"):
                routed_out = profiled_scatter_tokens(expert_output, metadata, "masked", book)
        with book.time("forward.shared_expert.grouped"):
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

    original_kt_run_moe = moe.KTTinyMoELayer._run_moe

    def timed_kt_run_moe(self: Any, x: torch.Tensor, *, static_routing: Any, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode != "contiguous":
            raise ValueError("backend=kt only supports moe_mode=contiguous")
        input_shape = x.shape
        with book.time("forward.moe.flatten"):
            flat = x.reshape(-1, self.config.hidden_size)
        with book.time("forward.router"):
            (topk_indices, routing_weights), logits = self._route(flat, static_routing)
        with book.time("forward.route_metadata"):
            metadata = moe.build_route_metadata(topk_indices, routing_weights, num_experts=self.config.num_experts, mode=mode)
        with book.time("forward.kt_moe.forward_sft"):
            routed_out = self.kt_moe(flat.contiguous(), topk_indices, routing_weights)
        with book.time("forward.moe.combine_shared_routed"):
            moe_out = routed_out
        return moe_out.reshape(input_shape), {
            "metadata": metadata,
            "logits": logits,
            "topk_indices": topk_indices,
            "routing_weights": routing_weights,
        }

    originals.append((moe.KTTinyMoELayer, "_run_moe", original_kt_run_moe))
    moe.KTTinyMoELayer._run_moe = timed_kt_run_moe

    original_model_forward = moe.TinyMoE.forward

    def timed_model_forward(self: Any, x: torch.Tensor | None = None, *, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, labels: torch.Tensor | None = None, static_routing: Any = None, mode: str = "contiguous", return_details: bool = False) -> Any:
        details = []
        with book.time("forward.embeddings"):
            hidden, token_api = self._prepare_hidden(x, input_ids, inputs_embeds)
        for layer_idx, layer in enumerate(self.layers):
            routing = moe._routing_for_layer(static_routing, layer_idx)
            if bool(getattr(self, "gradient_checkpointing", False)) and hidden.requires_grad and not return_details:
                def layer_forward(hidden_states: torch.Tensor, *, layer: Any = layer, routing: Any = routing) -> torch.Tensor:
                    result = layer(hidden_states, static_routing=routing, mode=mode, return_details=False)
                    assert isinstance(result, torch.Tensor)
                    return result

                hidden = checkpoint(layer_forward, hidden, use_reentrant=False)
            else:
                result = layer(hidden, static_routing=routing, mode=mode, return_details=return_details)
                if not return_details:
                    hidden = result
                    continue
                hidden, detail = result
                details.append(detail)
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
        "forward.kt_moe.forward_sft",
        "forward.pack_tokens",
        "forward.routed_expert.grouped",
        "forward.scatter_combine",
        "forward.shared_expert.grouped",
        "forward.moe.combine_shared_routed",
        "forward.moe.residual_add",
        "forward.final_norm",
    ]
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
        "backward.kt_moe.backward_sft",
    ]
    keys.extend(
        [
            "backward.moe.layernorm",
            "backward.attention.residual_add",
            "backward.attention.o_proj_base",
            "backward.attention.value_matmul",
            "backward.attention.softmax",
            "backward.attention.causal_mask",
            "backward.attention.scores_matmul",
            "backward.attention.v_proj_base",
            "backward.attention.k_proj_base",
            "backward.attention.q_proj_base",
            "backward.attention.layernorm",
        ]
    )
    return keys


def profile_moe(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.moe import (
        MICRO_MOE_CONFIG,
        make_static_routes,
        make_tiny_moe_pair,
    )

    clear(device)
    config = getattr(args, "_moe_config_override", MICRO_MOE_CONFIG)
    if not hasattr(args, "_moe_config_override"):
        batch_size = int(args.real_batch_size)
        seq_len = int(args.real_seq_len)
        config = replace(
            config,
            num_layers=max(1, min(int(args.real_profile_layers), int(config.num_layers))),
            batch_size=batch_size,
            seq_len=seq_len,
            logical_tokens=requested_tokens(args, batch_size * seq_len),
            lora_rank=int(args.real_lora_rank),
            lora_alpha=float(args.real_lora_alpha),
        )
    if int(config.num_shared_experts) != 0:
        config = replace(config, num_shared_experts=0)
        config_extra_shared_note = "forced_zero_for_kt_comparison"
    else:
        config_extra_shared_note = "already_zero"
    workload_name = str(getattr(args, "_workload_name_override", "m4_3_moe"))
    config_extra = dict(getattr(args, "_config_extra", {}))
    lora_dtype = profile_lora_dtype(args)
    target_selector = str(getattr(args, "target_modules", DEFAULT_TARGET_MODULES) or DEFAULT_TARGET_MODULES)
    offload_selector = str(getattr(args, "offload_modules", DEFAULT_MOE_OFFLOAD_MODULES) or DEFAULT_MOE_OFFLOAD_MODULES)
    target_groups = moe_selector_groups(target_selector, default=DEFAULT_TARGET_MODULES, purpose="target")
    if target_groups != {"routed_experts", "shared_experts"}:
        raise ValueError("toy MoE target_modules currently supports all/mlp only; use offload_modules for routed/shared CPU placement")
    config_extra["lora_dtype"] = str(lora_dtype)
    config_extra["target_modules"] = target_selector
    config_extra["offload_modules"] = offload_selector
    config_extra["shared_expert_policy"] = config_extra_shared_note
    config_extra["activation_recompute"] = bool(getattr(args, "activation_recompute", False))
    if is_kt_backend(args.backend):
        config_extra["kt_method"] = args.kt_method
        config_extra["kt_cpu_threads"] = int(args.kt_cpu_threads)
        config_extra["kt_threadpool_count"] = int(args.kt_threadpool_count)
        config_extra["kt_max_cache_depth"] = int(args.kt_max_cache_depth)
    config_extra["moe_target_groups"] = sorted(target_groups)
    config_extra["moe_offload_groups"] = sorted(moe_selector_groups(offload_selector, default=DEFAULT_MOE_OFFLOAD_MODULES, purpose="offload"))
    model, _, _, stats = make_tiny_moe_pair(
        config=config,
        seed=3,
        device=device,
        base_dtype=dtype,
        backend=args.backend,
        pin_memory=device.type == "cuda",
        offload_modules=offload_selector,
        lora_dtype=lora_dtype,
        precision=args.precision,
        kt_method=args.kt_method,
        kt_cpu_threads=args.kt_cpu_threads,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_max_cache_depth=args.kt_max_cache_depth,
        gradient_checkpointing=bool(getattr(args, "activation_recompute", False)),
    )
    set_profile_names(model)
    optimizer_params = model.kt_lora_parameters() if is_kt_backend(args.backend) else list(model.parameters())
    if not optimizer_params:
        raise RuntimeError("optimizer parameter list is empty")
    optimizer = torch.optim.AdamW(optimizer_params, lr=5e-3, weight_decay=0.0)
    static_routes = make_static_routes(config, device, pattern="balanced")

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        expected_tokens = int(config.batch_size) * int(config.seq_len)
        if int(config.logical_tokens) != expected_tokens:
            raise ValueError(
                f"MoE profile expects logical_tokens=batch_size*seq_len for transformer-shaped inputs; "
                f"got logical_tokens={config.logical_tokens}, batch_size={config.batch_size}, seq_len={config.seq_len}"
            )
        x = torch.randn(
            int(config.batch_size),
            int(config.seq_len),
            int(config.hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True,
        ) * 0.5
        target = torch.roll(x.detach().float(), shifts=1, dims=0) * 0.25
        return x, target

    def forward_fn(model_: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = model_(inputs_embeds=x, static_routing=static_routes, mode=args.moe_mode)
        if isinstance(y, dict):
            y = y["hidden_states"]
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
        stage_memory=book.memory_summary(),
        memory_attribution=memory_attribution_report(model, optimizer, device, book.saved_tensor_tracker),
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
    def post_optimizer_step() -> None:
        hook = getattr(model, "post_optimizer_step", None)
        if callable(hook) and getattr(model, "backend", None) == "kt":
            with book.time("optimizer.kt_lora_update"):
                hook()

    with profile_enabled(True):
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            batch, target = make_batch()
            loss = loss_fn(forward_fn(model, batch), target)
            loss.backward()
            optimizer.step()
            post_optimizer_step()
            sync(book.device)
        if reset_stats is not None:
            reset_stats()
        if book.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(book.device)
        if bool(getattr(args, "memory_attribution", False)):
            book.saved_tensor_tracker = SavedTensorMemoryTracker(model, book)
        book.clear()

        hooks = (
            torch.autograd.graph.saved_tensors_hooks(book.saved_tensor_tracker.pack, book.saved_tensor_tracker.unpack)
            if book.saved_tensor_tracker is not None
            else nullcontext()
        )
        with patch_base_dispatch(book):
            with hooks:
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
                        post_optimizer_step()
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
    stage_memory: dict[str, Any] | None = None,
    memory_attribution: dict[str, Any] | None = None,
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
    memory = memory_report(model, device)
    max_stage_peak = int((stage_memory or {}).get("max_stage_peak_bytes", 0))
    if max_stage_peak > 0:
        memory["gpu"]["stage_local_peak_hbm_bytes"] = max_stage_peak
        memory["gpu"]["peak_hbm_bytes"] = max(int(memory["gpu"]["peak_hbm_bytes"]), max_stage_peak)

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
        "memory": memory,
        "stage_memory": stage_memory or {"rows": []},
        "memory_attribution": memory_attribution or {"enabled": False, "categories": {"rows": []}, "saved_activations": {"rows": []}},
        "raw_seconds_per_step": raw_seconds_without_individual_calls(avg),
        "raw_dispatch_call_group_seconds_per_step": call_group_seconds_per_step(avg),
        "notes": [
            "Source-level rows are Python wall-clock range timings. In timing_mode=profile, inner ranges do not synchronize and should be treated as labels/CPU submission timing, not GPU execution timing.",
            "Use timing_mode=profile under Nsight Systems for real GPU bubble analysis; inner ranges are NVTX/record_function labels and do not force per-op CUDA synchronization.",
            "timing_mode=debug_sync is a debugging-only source coverage check that synchronizes every region and must not be used for performance claims.",
            "All explicit tensor ops in the instrumented toy forward/backward are assigned named ranges. Source-level residual rows are non-tensor-op overhead: Python dispatch, PyTorch autograd engine scheduling, CUDA launch latency, and synchronization/profiler overhead.",
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


def markdown(report: dict[str, Any]) -> str:
    def fmt_mib(value: int | float) -> str:
        return f"{float(value) / (1024.0 ** 2):.2f}"

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
        "This is not the GPU performance truth table. Use Nsight Systems plus `scripts/postprocess_nsys_lora.py` for kernel-busy, memcpy, and GPU no-kernel percentages.",
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
            "| Component | bytes | MiB |",
            "|---|---:|---:|",
            f"| peak_hbm | {mem['gpu']['peak_hbm_bytes']} | {fmt_mib(mem['gpu']['peak_hbm_bytes'])} |",
            f"| gpu_parameters | {mem['gpu']['parameter_bytes']} | {fmt_mib(mem['gpu']['parameter_bytes'])} |",
            f"| gpu_buffers | {mem['gpu']['buffer_bytes']} | {fmt_mib(mem['gpu']['buffer_bytes'])} |",
            f"| host_W | {mem['cpu']['host_w_bytes']} | {fmt_mib(mem['cpu']['host_w_bytes'])} |",
            f"| pinned_W | {mem['cpu']['pinned_w_bytes']} | {fmt_mib(mem['cpu']['pinned_w_bytes'])} |",
            f"| pinned_total | {mem['cpu']['pinned_total_bytes']} | {fmt_mib(mem['cpu']['pinned_total_bytes'])} |",
            "",
        ]
    )
    stage_memory = report.get("stage_memory", {})
    rows = stage_memory.get("rows") if isinstance(stage_memory, dict) else None
    if isinstance(rows, list) and rows:
        lines.extend(
            [
                "## Forward/Backward CUDA Allocator Memory",
                "",
                "| Stage | samples | allocated start MiB | allocated end MiB | allocated delta MiB | local peak MiB | local peak delta MiB | reserved start MiB | reserved end MiB | reserved delta MiB | global peak after MiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['name']} | "
                f"{row['samples']} | "
                f"{fmt_mib(row['avg_allocated_start_bytes'])} | "
                f"{fmt_mib(row['avg_allocated_end_bytes'])} | "
                f"{fmt_mib(row['avg_allocated_delta_bytes'])} | "
                f"{fmt_mib(row.get('avg_local_peak_bytes', row['max_global_peak_after_bytes']))} | "
                f"{fmt_mib(row.get('avg_local_peak_delta_bytes', 0))} | "
                f"{fmt_mib(row['avg_reserved_start_bytes'])} | "
                f"{fmt_mib(row['avg_reserved_end_bytes'])} | "
                f"{fmt_mib(row['avg_reserved_delta_bytes'])} | "
                f"{fmt_mib(row['max_global_peak_after_bytes'])} |"
            )
        lines.append("")
    attribution = report.get("memory_attribution", {})
    categories = attribution.get("categories", {}) if isinstance(attribution, dict) else {}
    category_rows = categories.get("rows") if isinstance(categories, dict) else None
    if isinstance(category_rows, list) and category_rows:
        lines.extend(
            [
                "## Fine-Grained Memory Attribution",
                "",
                "These rows are tensor-size accounting. Saved activation rows require `--memory-attribution` and are memory-only, not timing truth.",
                "",
                "| Category | Memory space | bytes | MiB | Accuracy |",
                "|---|---|---:|---:|---|",
            ]
        )
        for row in category_rows:
            value = int(row["bytes"])
            lines.append(f"| {row['category']} | {row['memory_space']} | {value} | {fmt_mib(value)} | {row['accuracy']} |")
        lines.append("")
    saved = attribution.get("saved_activations", {}) if isinstance(attribution, dict) else {}
    saved_rows = saved.get("rows") if isinstance(saved, dict) else None
    if isinstance(saved_rows, list) and saved_rows:
        lines.extend(
            [
                "## Saved Activation Memory by Semantic Owner",
                "",
                "| Owner | unique bytes | unique MiB | reference bytes | saves | unique tensors |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in saved_rows:
            value = int(row["unique_bytes"])
            lines.append(
                f"| {row['owner']} | {value} | {fmt_mib(value)} | "
                f"{int(row['reference_bytes'])} | {int(row['save_count'])} | {int(row['unique_tensor_count'])} |"
            )
        lines.append("")
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


def dense_config_from_metadata(metadata: dict[str, Any], args: argparse.Namespace) -> Any:
    from asym_gemm.training.dense import TinyDenseLLMConfig

    return TinyDenseLLMConfig(
        vocab_size=min(int(metadata["vocab_size"]), int(args.real_vocab_rows)),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=max(1, min(int(args.real_profile_layers), int(metadata["hf_num_hidden_layers"]))),
        num_heads=int(metadata["num_attention_heads"]),
        seq_len=int(args.real_seq_len),
        batch_size=int(args.real_batch_size),
        intermediate_size=int(metadata["intermediate_size"]),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
    )


def moe_config_from_metadata(metadata: dict[str, Any], args: argparse.Namespace) -> Any:
    from asym_gemm.training.moe import TinyMoEConfig

    return TinyMoEConfig(
        num_layers=max(1, min(int(args.real_profile_layers), int(metadata["hf_num_hidden_layers"]))),
        num_experts=int(metadata["num_experts"]),
        top_k=int(metadata["num_experts_per_tok"]),
        hidden_size=int(metadata["hidden_size"]),
        intermediate_size=int(metadata["moe_intermediate_size"]),
        logical_tokens=int(args.real_tokens or (int(args.real_seq_len) * int(args.real_batch_size))),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
        residual_scale=0.25,
        num_shared_experts=int(metadata["num_shared_experts"]),
        vocab_size=min(int(metadata["vocab_size"]), int(args.real_vocab_rows)),
        num_heads=int(metadata["num_attention_heads"]),
        batch_size=int(args.real_batch_size),
        seq_len=int(args.real_seq_len),
    )


def dense_14b_config(args: argparse.Namespace) -> Any:
    return dense_config_from_metadata(DENSE_14B_CONFIG, args)


def custom_dense_3b_config(args: argparse.Namespace) -> Any:
    return dense_config_from_metadata(CUSTOM_DENSE_3B_CONFIG, args)


def moe_604m_a38m_config(args: argparse.Namespace) -> Any:
    return moe_config_from_metadata(MOE_604M_A38M_CONFIG, args)


def moe_604m_a75m_config(args: argparse.Namespace) -> Any:
    return moe_config_from_metadata(MOE_604M_A75M_CONFIG, args)


def _metadata_config_extra(metadata: dict[str, Any], profiled_layers: int) -> dict[str, Any]:
    return {
        **metadata,
        "profiled_layers": int(profiled_layers),
        "weight_source": "random_config_matched",
    }


def profile_dense_metadata_workload(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    workload_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config = dense_config_from_metadata(metadata, args)
    args._dense_config_override = config
    args._workload_name_override = workload_name
    args._config_extra = _metadata_config_extra(metadata, config.num_layers)
    return profile_dense(args, device, dtype)


def profile_moe_metadata_workload(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    workload_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config = moe_config_from_metadata(metadata, args)
    args._moe_config_override = config
    args._workload_name_override = workload_name
    args._config_extra = _metadata_config_extra(metadata, config.num_layers)
    return profile_moe(args, device, dtype)


def run_workload(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    validate_backend_workload(args)
    simple_workloads: dict[str, Callable[[argparse.Namespace, torch.device, torch.dtype], dict[str, Any]]] = {
        "mlp": profile_mlp,
        "dense": profile_dense,
        "moe": profile_moe,
        "mlp_1b": profile_mlp_1b,
    }
    if args.workload in simple_workloads:
        return simple_workloads[args.workload](args, device, dtype)
    if args.workload in MM_CONFIGS:
        return profile_mm_lora(args, device, dtype, args.workload)
    if args.workload in MLP_CONFIGS:
        return profile_mlp_lora(args, device, dtype, args.workload)

    dense_metadata_workloads = {
        "dense_3b": CUSTOM_DENSE_3B_CONFIG,
        "dense_14b": DENSE_14B_CONFIG,
    }
    if args.workload in dense_metadata_workloads:
        return profile_dense_metadata_workload(
            args,
            device,
            dtype,
            workload_name=args.workload,
            metadata=dense_metadata_workloads[args.workload],
        )

    moe_metadata_workloads = {
        "moe-604m-a75m": MOE_604M_A75M_CONFIG,
        "moe-604m-a38m": MOE_604M_A38M_CONFIG,
    }
    if args.workload in moe_metadata_workloads:
        return profile_moe_metadata_workload(
            args,
            device,
            dtype,
            workload_name=args.workload,
            metadata=moe_metadata_workloads[args.workload],
        )
    raise AssertionError(args.workload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=WORKLOAD_CHOICES,
        default="mlp",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="asym")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
    parser.add_argument(
        "--target-preset",
        "--dense-target-mode",
        dest="target_preset",
        choices=DENSE_TARGET_MODES,
        default=DEFAULT_DENSE_TARGET_MODE,
        help="LoRA target preset for toy/HF-style projections. Default adapts all known target projections.",
    )
    parser.add_argument(
        "--target-modules",
        default=DEFAULT_TARGET_MODULES,
        help="Target module selector or comma list. Examples: all, mlp, attention, q_proj,v_proj.",
    )
    parser.add_argument(
        "--offload-modules",
        default=None,
        help="CPU/AsymGEMM base offload selector or comma list. Dense default is mlp; MoE default is routed_experts.",
    )
    parser.add_argument("--profile-layers", "--real-profile-layers", dest="real_profile_layers", metavar="N", type=int, default=1)
    parser.add_argument("--batch-size", "--real-batch-size", dest="real_batch_size", metavar="N", type=int, default=DEFAULT_LORA_BATCH_SIZE)
    parser.add_argument("--seq-len", "--real-seq-len", dest="real_seq_len", metavar="N", type=int, default=DEFAULT_LORA_SEQ_LEN)
    parser.add_argument("--tokens", "--real-tokens", dest="real_tokens", metavar="N", type=int, default=0)
    parser.add_argument(
        "--hidden-dim",
        "--real-hidden-dim",
        dest="real_hidden_dim",
        metavar="N",
        type=int,
        default=DEFAULT_LORA_HIDDEN_DIM,
        help="Model hidden width for synthetic mm_3b/mlp_3b LoRA profiles.",
    )
    parser.add_argument(
        "--mlp-intermediate-dim",
        "--real-mlp-intermediate-dim",
        dest="real_mlp_intermediate_dim",
        metavar="N",
        type=int,
        default=0,
        help="Intermediate width for synthetic mlp_3b. Default is hidden_dim * mlp_expansion.",
    )
    parser.add_argument(
        "--mlp-expansion",
        "--real-mlp-expansion",
        dest="real_mlp_expansion",
        metavar="N",
        type=int,
        default=DEFAULT_LORA_MLP_EXPANSION,
        help="MLP expansion used when --mlp-intermediate-dim is not set.",
    )
    parser.add_argument("--lora-rank", "--real-lora-rank", dest="real_lora_rank", metavar="N", type=int, default=64)
    parser.add_argument("--lora-alpha", "--real-lora-alpha", dest="real_lora_alpha", metavar="FLOAT", type=float, default=128.0)
    parser.add_argument("--lora-dtype", choices=LORA_DTYPE_CHOICES, default="bf16")
    parser.add_argument("--vocab-rows", "--real-vocab-rows", dest="real_vocab_rows", metavar="N", type=int, default=4096)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "fp4"])
    parser.add_argument("--kt-method", default="AMXBF16_SFT", choices=KT_METHOD_CHOICES)
    parser.add_argument("--kt-cpu-threads", type=int, default=0, help="KT CPUInfer threads. 0 selects all available CPUs.")
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-max-cache-depth", type=int, default=1)
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
    parser.add_argument(
        "--memory-attribution",
        action="store_true",
        help="Enable saved-tensor hooks for fine-grained activation memory attribution. This is memory-only and should not be used for timing claims.",
    )
    parser.add_argument(
        "--activation-recompute",
        action="store_true",
        help="Enable layer-level activation recomputation with torch.utils.checkpoint during training profiles.",
    )
    parsed = parser.parse_args()
    parsed.dense_target_mode = parsed.target_preset
    try:
        validate_backend_workload(parsed)
    except ValueError as exc:
        parser.error(str(exc))
    return parsed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    def run() -> dict[str, Any]:
        return run_workload(args, device, dtype)

    try:
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
    except KTBackendUnavailable as exc:
        raise SystemExit(str(exc)) from None
    write_report(report, args.output_dir)
    print(json.dumps({"workload": report["workload"], "step_ms": report["step"]["total_milliseconds"]}, indent=2))


if __name__ == "__main__":
    main()
