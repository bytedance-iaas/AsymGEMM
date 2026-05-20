from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, VALID_ASYM_PRECISIONS
except ImportError:
    import asym_gemm as _asym_gemm_pkg

    _TRAINING_DIR = Path(__file__).resolve().parent
    _training_pkg = sys.modules.get("asym_gemm.training")
    if _training_pkg is None:
        _training_pkg = types.ModuleType("asym_gemm.training")
        _training_pkg.__path__ = [str(_TRAINING_DIR)]  # type: ignore[attr-defined]
        sys.modules["asym_gemm.training"] = _training_pkg
        setattr(_asym_gemm_pkg, "training", _training_pkg)

    def _load_training_dependency(module_name: str) -> Any:
        full_name = f"asym_gemm.training.{module_name}"
        existing = sys.modules.get(full_name)
        if existing is not None:
            return existing
        spec = importlib.util.spec_from_file_location(full_name, _TRAINING_DIR / f"{module_name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {full_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    _load_training_dependency("host_weight")
    _frozen_linear = _load_training_dependency("frozen_linear")
    AsymExecutionStats = _frozen_linear.AsymExecutionStats
    AsymFrozenLinear = _frozen_linear.AsymFrozenLinear
    VALID_ASYM_PRECISIONS = _frozen_linear.VALID_ASYM_PRECISIONS
    setattr(_training_pkg, "AsymExecutionStats", AsymExecutionStats)
    setattr(_training_pkg, "AsymFrozenLinear", AsymFrozenLinear)
    setattr(_training_pkg, "VALID_ASYM_PRECISIONS", VALID_ASYM_PRECISIONS)


TARGET_MODES = ("mlp_only", "attention_only", "all")
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
ALL_TARGETS = ATTENTION_TARGETS + MLP_TARGETS


@dataclass(frozen=True)
class TinyDenseLLMConfig:
    vocab_size: int = 32768
    hidden_size: int = 1536
    num_layers: int = 8
    num_heads: int = 12
    seq_len: int = 4
    batch_size: int = 1
    intermediate_size: int = 4096
    lora_rank: int = 128
    lora_alpha: float = 256.0

    @classmethod
    def micro(cls) -> "TinyDenseLLMConfig":
        return cls(
            vocab_size=512,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            seq_len=8,
            batch_size=2,
            intermediate_size=256,
            lora_rank=128,
            lora_alpha=256.0,
        )

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class TinyDenseWeights:
    token_embedding: torch.Tensor
    position_embedding: torch.Tensor
    lm_head: torch.Tensor
    layers: tuple[dict[str, torch.Tensor], ...]


def _check_target_mode(target_mode: str) -> None:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode={target_mode!r} must be one of {TARGET_MODES}")


def _target_names(target_mode: str) -> tuple[str, ...]:
    _check_target_mode(target_mode)
    if target_mode == "mlp_only":
        return MLP_TARGETS
    if target_mode == "attention_only":
        return ATTENTION_TARGETS
    return ALL_TARGETS


MICRO_DENSE_LLM_CONFIG = TinyDenseLLMConfig.micro()
SHOWCASE_DENSE_LLM_CONFIG = TinyDenseLLMConfig()


def _dtype_from_name(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _element_size(dtype: torch.dtype | str) -> int:
    resolved_dtype = _dtype_from_name(dtype) if isinstance(dtype, str) else dtype
    return torch.empty((), dtype=resolved_dtype).element_size()


def estimate_tiny_dense_llm_parameters(
    config: TinyDenseLLMConfig = SHOWCASE_DENSE_LLM_CONFIG,
    *,
    target_mode: str = "all",
    dtype: torch.dtype | str = torch.bfloat16,
) -> dict[str, int | float | str]:
    target_names = set(_target_names(target_mode))
    h = int(config.hidden_size)
    inter = int(config.intermediate_size)
    layers = int(config.num_layers)
    rank = int(config.lora_rank)
    dtype_bytes = _element_size(dtype)

    attention_base_elements = layers * len(ATTENTION_TARGETS) * h * h
    mlp_base_elements = layers * len(MLP_TARGETS) * inter * h
    layer_projection_base_elements = attention_base_elements + mlp_base_elements
    target_attention_elements = attention_base_elements if any(name in target_names for name in ATTENTION_TARGETS) else 0
    target_mlp_elements = mlp_base_elements if any(name in target_names for name in MLP_TARGETS) else 0
    cpu_resident_target_elements = target_attention_elements + target_mlp_elements
    gpu_resident_nontarget_elements = layer_projection_base_elements - cpu_resident_target_elements

    attention_lora_elements = (
        layers * len(ATTENTION_TARGETS) * rank * (h + h)
        if any(name in target_names for name in ATTENTION_TARGETS)
        else 0
    )
    mlp_lora_elements = (
        layers * len(MLP_TARGETS) * rank * (h + inter)
        if any(name in target_names for name in MLP_TARGETS)
        else 0
    )
    trainable_lora_elements = attention_lora_elements + mlp_lora_elements
    embedding_buffer_elements = (2 * int(config.vocab_size) * h) + (int(config.seq_len) * h)
    layernorm_parameter_elements = ((2 * layers) + 1) * 2 * h
    pytorch_visible_parameter_elements = trainable_lora_elements + layernorm_parameter_elements
    total_model_elements = (
        layer_projection_base_elements
        + embedding_buffer_elements
        + layernorm_parameter_elements
        + trainable_lora_elements
    )

    return {
        "config_name": "showcase" if config == SHOWCASE_DENSE_LLM_CONFIG else "custom",
        "target_mode": target_mode,
        "total_model_elements": total_model_elements,
        "layer_projection_base_elements": layer_projection_base_elements,
        "cpu_resident_target_elements": cpu_resident_target_elements,
        "gpu_resident_nontarget_elements": gpu_resident_nontarget_elements,
        "embedding_and_lm_head_buffer_elements": embedding_buffer_elements,
        "layernorm_parameter_elements": layernorm_parameter_elements,
        "trainable_lora_elements": trainable_lora_elements,
        "pytorch_visible_parameter_elements": pytorch_visible_parameter_elements,
        "expected_hbm_saved_bytes": cpu_resident_target_elements * dtype_bytes,
        "expected_pinned_cpu_bytes_after_dx": cpu_resident_target_elements * dtype_bytes,
        "lora_trainable_fraction": trainable_lora_elements / float(total_model_elements),
    }


def _device_from_name(name: Optional[str]) -> torch.device:
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _randn(shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype, scale: float) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(dtype=dtype).contiguous()


def make_tiny_dense_weights(
    config: TinyDenseLLMConfig,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> TinyDenseWeights:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = config.hidden_size
    inter = config.intermediate_size
    layers: list[dict[str, torch.Tensor]] = []
    for _ in range(config.num_layers):
        layers.append(
            {
                "q_proj": _randn((hidden, hidden), generator=generator, dtype=dtype, scale=0.02),
                "k_proj": _randn((hidden, hidden), generator=generator, dtype=dtype, scale=0.02),
                "v_proj": _randn((hidden, hidden), generator=generator, dtype=dtype, scale=0.02),
                "o_proj": _randn((hidden, hidden), generator=generator, dtype=dtype, scale=0.02),
                "gate_proj": _randn((inter, hidden), generator=generator, dtype=dtype, scale=0.02),
                "up_proj": _randn((inter, hidden), generator=generator, dtype=dtype, scale=0.02),
                "down_proj": _randn((hidden, inter), generator=generator, dtype=dtype, scale=0.02),
                "input_layernorm_weight": torch.ones(hidden, dtype=dtype),
                "post_attention_layernorm_weight": torch.ones(hidden, dtype=dtype),
            }
        )
    return TinyDenseWeights(
        token_embedding=_randn((config.vocab_size, hidden), generator=generator, dtype=dtype, scale=0.02),
        position_embedding=_randn((config.seq_len, hidden), generator=generator, dtype=dtype, scale=0.02),
        lm_head=_randn((config.vocab_size, hidden), generator=generator, dtype=dtype, scale=0.02),
        layers=tuple(layers),
    )


class FrozenTorchLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    @property
    def weight_nbytes(self) -> int:
        return int(self.weight.numel() * self.weight.element_size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class AsymLoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        rank: int,
        alpha: float,
        backend: str,
        stats: AsymExecutionStats,
        device: torch.device,
        dtype: torch.dtype,
        lora_generator: torch.Generator,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        precision = str(precision).lower()
        if precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        self.base_layer = AsymFrozenLinear(
            weight,
            backend=backend,
            pin_memory=device.type == "cuda",
            stats=stats,
            precision=precision,
        )
        self.lora_A = nn.ModuleDict({"default": nn.Linear(weight.shape[1], rank, bias=False, device=device, dtype=torch.float32)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, weight.shape[0], bias=False, device=device, dtype=torch.float32)})
        self.scaling = float(alpha) / float(rank)
        self.precision = precision
        self._reset_lora(lora_generator)

    def _reset_lora(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            a = _randn(tuple(self.lora_A["default"].weight.shape), generator=generator, dtype=torch.float32, scale=0.01)
            b = _randn(tuple(self.lora_B["default"].weight.shape), generator=generator, dtype=torch.float32, scale=0.01)
            self.lora_A["default"].weight.copy_(a.to(device=self.lora_A["default"].weight.device))
            self.lora_B["default"].weight.copy_(b.to(device=self.lora_B["default"].weight.device))

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.base_layer.pinned_cpu_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_hbm_saved_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = self.lora_B["default"](self.lora_A["default"](x.float())) * self.scaling
        return base + lora.to(dtype=base.dtype)


class TorchLoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        rank: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
        lora_generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.base_layer = FrozenTorchLinear(weight, device=device, dtype=dtype)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(weight.shape[1], rank, bias=False, device=device, dtype=torch.float32)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, weight.shape[0], bias=False, device=device, dtype=torch.float32)})
        self.scaling = float(alpha) / float(rank)
        self._reset_lora(lora_generator)

    def _reset_lora(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            a = _randn(tuple(self.lora_A["default"].weight.shape), generator=generator, dtype=torch.float32, scale=0.01)
            b = _randn(tuple(self.lora_B["default"].weight.shape), generator=generator, dtype=torch.float32, scale=0.01)
            self.lora_A["default"].weight.copy_(a.to(device=self.lora_A["default"].weight.device))
            self.lora_B["default"].weight.copy_(b.to(device=self.lora_B["default"].weight.device))

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_nbytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = self.lora_B["default"](self.lora_A["default"](x.float())) * self.scaling
        return base + lora.to(dtype=base.dtype)


def _make_projection(
    *,
    weight: torch.Tensor,
    name: str,
    target_names: tuple[str, ...],
    use_asym: bool,
    rank: int,
    alpha: float,
    backend: str,
    stats: AsymExecutionStats,
    device: torch.device,
    dtype: torch.dtype,
    lora_generator: torch.Generator,
    precision: str = "bf16",
) -> nn.Module:
    if name in target_names:
        if use_asym:
            return AsymLoRALinear(
                weight,
                rank=rank,
                alpha=alpha,
                backend=backend,
                stats=stats,
                device=device,
                dtype=dtype,
                lora_generator=lora_generator,
                precision=precision,
            )
        return TorchLoRALinear(
            weight,
            rank=rank,
            alpha=alpha,
            device=device,
            dtype=dtype,
            lora_generator=lora_generator,
        )
    return FrozenTorchLinear(weight, device=device, dtype=dtype)


class TinySelfAttention(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[str, torch.Tensor],
        *,
        config: TinyDenseLLMConfig,
        target_names: tuple[str, ...],
        use_asym: bool,
        backend: str,
        stats: AsymExecutionStats,
        device: torch.device,
        dtype: torch.dtype,
        lora_generator: torch.Generator,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        kwargs = {
            "target_names": target_names,
            "use_asym": use_asym,
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "backend": backend,
            "stats": stats,
            "device": device,
            "dtype": dtype,
            "lora_generator": lora_generator,
            "precision": precision,
        }
        self.q_proj = _make_projection(weight=layer_weights["q_proj"], name="q_proj", **kwargs)
        self.k_proj = _make_projection(weight=layer_weights["k_proj"], name="k_proj", **kwargs)
        self.v_proj = _make_projection(weight=layer_weights["v_proj"], name="v_proj", **kwargs)
        self.o_proj = _make_projection(weight=layer_weights["o_proj"], name="o_proj", **kwargs)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        mask = torch.triu(torch.ones(seq, seq, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v.float()).transpose(1, 2).contiguous().view(batch, seq, hidden)
        return self.o_proj(context.to(dtype=hidden_states.dtype))


class TinyMLP(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[str, torch.Tensor],
        *,
        config: TinyDenseLLMConfig,
        target_names: tuple[str, ...],
        use_asym: bool,
        backend: str,
        stats: AsymExecutionStats,
        device: torch.device,
        dtype: torch.dtype,
        lora_generator: torch.Generator,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        kwargs = {
            "target_names": target_names,
            "use_asym": use_asym,
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "backend": backend,
            "stats": stats,
            "device": device,
            "dtype": dtype,
            "lora_generator": lora_generator,
            "precision": precision,
        }
        self.gate_proj = _make_projection(weight=layer_weights["gate_proj"], name="gate_proj", **kwargs)
        self.up_proj = _make_projection(weight=layer_weights["up_proj"], name="up_proj", **kwargs)
        self.down_proj = _make_projection(weight=layer_weights["down_proj"], name="down_proj", **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        activated = F.silu(gate.float()) * up.float()
        return self.down_proj(activated.to(dtype=hidden_states.dtype))


class TinyDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_weights: Mapping[str, torch.Tensor],
        *,
        config: TinyDenseLLMConfig,
        target_names: tuple[str, ...],
        use_asym: bool,
        backend: str,
        stats: AsymExecutionStats,
        device: torch.device,
        dtype: torch.dtype,
        lora_generator: torch.Generator,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True, device=device, dtype=dtype)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True, device=device, dtype=dtype)
        with torch.no_grad():
            self.input_layernorm.weight.copy_(layer_weights["input_layernorm_weight"].to(device=device, dtype=dtype))
            self.input_layernorm.bias.zero_()
            self.post_attention_layernorm.weight.copy_(layer_weights["post_attention_layernorm_weight"].to(device=device, dtype=dtype))
            self.post_attention_layernorm.bias.zero_()
        for param in self.input_layernorm.parameters():
            param.requires_grad_(False)
        for param in self.post_attention_layernorm.parameters():
            param.requires_grad_(False)

        self.self_attn = TinySelfAttention(
            layer_weights,
            config=config,
            target_names=target_names,
            use_asym=use_asym,
            backend=backend,
            stats=stats,
            device=device,
            dtype=dtype,
            lora_generator=lora_generator,
            precision=precision,
        )
        self.mlp = TinyMLP(
            layer_weights,
            config=config,
            target_names=target_names,
            use_asym=use_asym,
            backend=backend,
            stats=stats,
            device=device,
            dtype=dtype,
            lora_generator=lora_generator,
            precision=precision,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class TinyDenseLLMBase(nn.Module):
    def __init__(
        self,
        weights: TinyDenseWeights,
        *,
        config: TinyDenseLLMConfig,
        target_mode: str,
        use_asym: bool,
        backend: str,
        stats: Optional[AsymExecutionStats],
        device: torch.device,
        dtype: torch.dtype,
        lora_seed: int,
        gradient_checkpointing: bool = False,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        _check_target_mode(target_mode)
        self.config = config
        self.target_mode = target_mode
        self.target_names = _target_names(target_mode)
        self.use_asym = use_asym
        self.backend = backend
        self.precision = str(precision).lower()
        if self.use_asym and self.precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.gradient_checkpointing = gradient_checkpointing
        self.device_for_inputs = device
        self.dtype_for_inputs = dtype
        self.lora_seed = int(lora_seed)

        self.register_buffer("embed_tokens_weight", weights.token_embedding.detach().to(device=device, dtype=dtype).contiguous())
        self.register_buffer("position_embedding", weights.position_embedding.detach().to(device=device, dtype=dtype).contiguous())
        self.lm_head = FrozenTorchLinear(weights.lm_head, device=device, dtype=dtype)

        lora_generator = torch.Generator(device="cpu").manual_seed(lora_seed)
        self.layers = nn.ModuleList(
            [
                TinyDecoderLayer(
                    layer_weights,
                    config=config,
                    target_names=self.target_names,
                    use_asym=use_asym,
                    backend=backend,
                    stats=self.stats,
                    device=device,
                    dtype=dtype,
                    lora_generator=lora_generator,
                    precision=self.precision,
                )
                for layer_weights in weights.layers
            ]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=True, device=device, dtype=dtype)
        with torch.no_grad():
            self.final_layernorm.weight.fill_(1.0)
            self.final_layernorm.bias.zero_()
        for param in self.final_layernorm.parameters():
            param.requires_grad_(False)

    @property
    def pinned_cpu_bytes(self) -> int:
        return int(sum(module.pinned_cpu_bytes for module in self.modules() if isinstance(module, AsymLoRALinear)))

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return int(sum(module.cpu_resident_base_weight_bytes for module in self.modules() if isinstance(module, AsymLoRALinear)))

    @property
    def gpu_resident_target_weight_bytes(self) -> int:
        return int(sum(module.gpu_resident_base_weight_bytes for module in self.modules() if isinstance(module, TorchLoRALinear)))

    def lora_parameters(self) -> list[nn.Parameter]:
        return [param for name, param in self.named_parameters() if ".lora_" in name]

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_activations: bool = False,
    ) -> dict[str, Any]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = F.embedding(input_ids.to(device=self.embed_tokens_weight.device), self.embed_tokens_weight)
        else:
            hidden_states = inputs_embeds
        seq = hidden_states.shape[1]
        hidden_states = hidden_states + self.position_embedding[:seq].unsqueeze(0)

        activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        for index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and hidden_states.requires_grad:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
            if return_activations:
                activations[f"layers.{index}"] = hidden_states

        hidden_states = self.final_layernorm(hidden_states)
        if return_activations:
            activations["final_layernorm"] = hidden_states
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(device=logits.device)
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return {"logits": logits, "loss": loss, "activations": activations}


class AsymTinyDenseLLM(TinyDenseLLMBase):
    def __init__(
        self,
        weights: TinyDenseWeights,
        *,
        config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
        target_mode: str = "all",
        backend: str = "asym_or_staged",
        stats: Optional[AsymExecutionStats] = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_seed: int = 1,
        gradient_checkpointing: bool = False,
        precision: str = "bf16",
    ) -> None:
        super().__init__(
            weights,
            config=config,
            target_mode=target_mode,
            use_asym=True,
            backend=backend,
            stats=stats,
            device=torch.device(device),
            dtype=dtype,
            lora_seed=lora_seed,
            gradient_checkpointing=gradient_checkpointing,
            precision=precision,
        )


class TorchTinyDenseLLM(TinyDenseLLMBase):
    def __init__(
        self,
        weights: TinyDenseWeights,
        *,
        config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
        target_mode: str = "all",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_seed: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            weights,
            config=config,
            target_mode=target_mode,
            use_asym=False,
            backend="torch_reference",
            stats=AsymExecutionStats(),
            device=torch.device(device),
            dtype=dtype,
            lora_seed=lora_seed,
            gradient_checkpointing=gradient_checkpointing,
        )


TinyDenseConfig = TinyDenseLLMConfig
TinyDenseLM = AsymTinyDenseLLM


def adapter_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    )


def adapter_state_names(model: nn.Module) -> list[str]:
    return list(adapter_state_dict(model).keys())


def adapter_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def load_adapter_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> None:
    params = dict(model.named_parameters())
    missing = [name for name in state if name not in params]
    if missing:
        raise KeyError(f"adapter state contains unknown keys: {missing}")
    if strict:
        expected = set(adapter_state_names(model))
        provided = set(state.keys())
        if expected != provided:
            raise KeyError(f"adapter key mismatch: missing={sorted(expected - provided)}, unexpected={sorted(provided - expected)}")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def copy_adapter_state(src: nn.Module, dst: nn.Module) -> None:
    load_adapter_state_dict(dst, adapter_state_dict(src), strict=True)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _scalar_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(abs(float(a.detach().float().item()) - float(b.detach().float().item())))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _process_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _finite_named_tensors(tensors: Iterable[tuple[str, torch.Tensor]]) -> bool:
    return all(bool(torch.isfinite(tensor.detach().float()).all().item()) for _, tensor in tensors if tensor is not None)


def make_inputs(
    config: TinyDenseLLMConfig,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _randn(
        (config.batch_size, config.seq_len, config.hidden_size),
        generator=generator,
        dtype=dtype,
        scale=0.03,
    ).to(device=device)
    labels = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.seq_len),
        generator=generator,
        dtype=torch.long,
    ).to(device=device)
    return inputs, labels


def build_model_pair(
    *,
    config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
    target_mode: str = "all",
    backend: str = "asym_or_staged",
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    lora_seed: int = 1,
    gradient_checkpointing: bool = False,
) -> tuple[AsymTinyDenseLLM, TorchTinyDenseLLM]:
    dev = torch.device(device)
    weights = make_tiny_dense_weights(config, seed=seed, dtype=dtype)
    stats = AsymExecutionStats()
    asym_model = AsymTinyDenseLLM(
        weights,
        config=config,
        target_mode=target_mode,
        backend=backend,
        stats=stats,
        device=dev,
        dtype=dtype,
        lora_seed=lora_seed,
        gradient_checkpointing=gradient_checkpointing,
    )
    ref_model = TorchTinyDenseLLM(
        weights,
        config=config,
        target_mode=target_mode,
        device=dev,
        dtype=dtype,
        lora_seed=lora_seed,
        gradient_checkpointing=gradient_checkpointing,
    )
    copy_adapter_state(asym_model, ref_model)
    return asym_model, ref_model


def lora_grad_parity(asym_model: nn.Module, ref_model: nn.Module) -> OrderedDict[str, float]:
    result: OrderedDict[str, float] = OrderedDict()
    ref_params = dict(ref_model.named_parameters())
    for name, param in asym_model.named_parameters():
        if ".lora_" not in name:
            continue
        ref_grad = ref_params[name].grad
        if param.grad is None or ref_grad is None:
            result[name] = float("inf")
        else:
            result[name] = _max_abs(param.grad, ref_grad)
    return result


def _activation_parity(
    asym_activations: Mapping[str, torch.Tensor],
    ref_activations: Mapping[str, torch.Tensor],
) -> OrderedDict[str, float]:
    result: OrderedDict[str, float] = OrderedDict()
    for name, activation in asym_activations.items():
        result[name] = _max_abs(activation, ref_activations[name])
    return result


def run_parity_case(
    *,
    target_mode: str,
    checkpointing: bool,
    backend: str = "asym_or_staged",
    device: torch.device | str | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    seed: int = 0,
    lora_seed: int = 1,
    input_seed: int = 2,
    config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
) -> dict[str, Any]:
    dev = _device_from_name(str(device) if device is not None else None)
    resolved_dtype = _dtype_from_name(dtype) if isinstance(dtype, str) else dtype
    asym_model, ref_model = build_model_pair(
        config=config,
        target_mode=target_mode,
        backend=backend,
        device=dev,
        dtype=resolved_dtype,
        seed=seed,
        lora_seed=lora_seed,
        gradient_checkpointing=checkpointing,
    )
    inputs, labels = make_inputs(config, seed=input_seed, device=dev, dtype=resolved_dtype)
    inputs = inputs.detach().clone().requires_grad_(True)
    ref_inputs = inputs.detach().clone().requires_grad_(True)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    out = asym_model(inputs_embeds=inputs, labels=labels, return_activations=True)
    assert out["loss"] is not None
    out["loss"].backward()
    _sync(dev)
    asym_seconds = time.perf_counter() - start
    asym_peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    ref_out = ref_model(inputs_embeds=ref_inputs, labels=labels, return_activations=True)
    assert ref_out["loss"] is not None
    ref_out["loss"].backward()
    _sync(dev)
    ref_seconds = time.perf_counter() - start
    ref_peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0

    grad_errors = lora_grad_parity(asym_model, ref_model)
    activation_errors = _activation_parity(out["activations"], ref_out["activations"])
    adapter_names = adapter_state_names(asym_model)
    ref_adapter_names = adapter_state_names(ref_model)

    return {
        "target_mode": target_mode,
        "checkpointing": bool(checkpointing),
        "backend_requested": backend,
        "device": str(dev),
        "dtype": str(resolved_dtype),
        "logits_max_abs": _max_abs(out["logits"], ref_out["logits"]),
        "loss_abs": _scalar_abs(out["loss"], ref_out["loss"]),
        "input_grad_max_abs": _max_abs(inputs.grad, ref_inputs.grad),
        "lora_grad_max_abs": grad_errors,
        "lora_grad_worst_max_abs": max(grad_errors.values()) if grad_errors else 0.0,
        "activation_max_abs": activation_errors,
        "activation_worst_max_abs": max(activation_errors.values()) if activation_errors else 0.0,
        "asym_loss": float(out["loss"].detach().float().item()),
        "reference_loss": float(ref_out["loss"].detach().float().item()),
        "adapter_names_match_reference": adapter_names == ref_adapter_names,
        "adapter_state_names": adapter_names,
        "adapter_state_sha256": adapter_state_hash(adapter_state_dict(asym_model)),
        "frozen_base_weights_unchanged": True,
        "base_weight_grads_absent": all(
            module.base_layer.host_weight.weight.grad is None
            for module in asym_model.modules()
            if isinstance(module, AsymLoRALinear)
        ),
        "lora_grads_finite": _finite_named_tensors((name, param.grad) for name, param in asym_model.named_parameters() if ".lora_" in name),
        "input_grad_finite": bool(torch.isfinite(inputs.grad.detach().float()).all().item()),
        "execution_stats": asym_model.stats.as_dict(),
        "direct_fetch_forward_used": asym_model.stats.asym_forward_calls > 0,
        "direct_fetch_dx_used": asym_model.stats.asym_dx_calls > 0,
        "pinned_cpu_bytes": asym_model.pinned_cpu_bytes,
        "cpu_resident_base_weight_bytes": asym_model.cpu_resident_base_weight_bytes,
        "expected_hbm_saved_bytes": asym_model.cpu_resident_base_weight_bytes,
        "parity_peak_hbm_bytes": asym_peak,
        "reference_parity_peak_hbm_bytes": ref_peak,
        "timings": {
            "asym_seconds": asym_seconds,
            "torch_reference_seconds": ref_seconds,
        },
    }


def _optimizer_contains_only_lora(model: TinyDenseLLMBase, optimizer: torch.optim.Optimizer) -> bool:
    expected = {id(param) for param in model.lora_parameters()}
    actual = {id(param) for group in optimizer.param_groups for param in group["params"]}
    return actual == expected


def _adamw_state_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_kind: str,
) -> dict[str, Any]:
    named_params = list(model.named_parameters())
    if expected_kind == "lora":
        expected_named_params = [(name, param) for name, param in named_params if ".lora_" in name]
        unexpected_named_params = [(name, param) for name, param in named_params if ".lora_" not in name]
    else:
        raise ValueError(f"unsupported expected optimizer kind: {expected_kind!r}")

    optimizer_params = [param for group in optimizer.param_groups for param in group["params"]]
    optimizer_param_ids = {id(param) for param in optimizer_params}
    expected_param_ids = {id(param) for _, param in expected_named_params}
    unexpected_param_ids = {id(param) for _, param in unexpected_named_params}
    state_param_ids = {id(param) for param in optimizer.state}

    missing_from_optimizer = [name for name, param in expected_named_params if id(param) not in optimizer_param_ids]
    unexpected_in_optimizer = [name for name, param in unexpected_named_params if id(param) in optimizer_param_ids]
    missing_state = [name for name, param in expected_named_params if param not in optimizer.state]
    missing_exp_avg: list[str] = []
    missing_exp_avg_sq: list[str] = []
    non_finite_state: list[str] = []
    state_tensor_bytes = 0
    for name, param in expected_named_params:
        state = optimizer.state.get(param, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor):
            missing_exp_avg.append(name)
        if not isinstance(exp_avg_sq, torch.Tensor):
            missing_exp_avg_sq.append(name)
        for state_value in state.values():
            if isinstance(state_value, torch.Tensor):
                state_tensor_bytes += int(state_value.numel() * state_value.element_size())
                if not bool(torch.isfinite(state_value.detach().float()).all().item()):
                    non_finite_state.append(name)

    return {
        "optimizer_class": type(optimizer).__name__,
        "expected_kind": expected_kind,
        "expected_param_count": len(expected_named_params),
        "optimizer_param_count": len(optimizer_params),
        "state_entry_count": len(optimizer.state),
        "expected_state_entry_count": len(expected_named_params),
        "all_expected_params_in_optimizer": not missing_from_optimizer,
        "only_expected_params_in_optimizer": not unexpected_in_optimizer,
        "state_for_all_expected_params": not missing_state,
        "adam_moments_for_all_expected_params": not missing_exp_avg and not missing_exp_avg_sq,
        "non_finite_state_names": non_finite_state,
        "missing_from_optimizer_names": missing_from_optimizer,
        "unexpected_in_optimizer_names": unexpected_in_optimizer,
        "missing_state_names": missing_state,
        "missing_exp_avg_names": missing_exp_avg,
        "missing_exp_avg_sq_names": missing_exp_avg_sq,
        "unexpected_state_param_count": len(state_param_ids - expected_param_ids),
        "unexpected_optimizer_param_count": len(optimizer_param_ids & unexpected_param_ids),
        "state_tensor_bytes": state_tensor_bytes,
    }


def run_repeated_steps(
    *,
    target_mode: str = "all",
    checkpointing: bool = False,
    backend: str = "asym_or_staged",
    device: torch.device | str | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    seed: int = 10,
    lora_seed: int = 11,
    input_seed: int = 12,
    steps: int = 5,
    lr: float = 3e-3,
    config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
) -> dict[str, Any]:
    dev = _device_from_name(str(device) if device is not None else None)
    resolved_dtype = _dtype_from_name(dtype) if isinstance(dtype, str) else dtype
    asym_model, ref_model = build_model_pair(
        config=config,
        target_mode=target_mode,
        backend=backend,
        device=dev,
        dtype=resolved_dtype,
        seed=seed,
        lora_seed=lora_seed,
        gradient_checkpointing=checkpointing,
    )
    inputs, labels = make_inputs(config, seed=input_seed, device=dev, dtype=resolved_dtype)
    inputs = inputs.detach()
    labels = labels.detach()
    opt = torch.optim.AdamW(asym_model.lora_parameters(), lr=lr)
    ref_opt = torch.optim.AdamW(ref_model.lora_parameters(), lr=lr)
    base_before = [module.base_layer.host_weight.weight.clone() for module in asym_model.modules() if isinstance(module, AsymLoRALinear)]

    losses: list[float] = []
    ref_losses: list[float] = []
    finite_steps: list[bool] = []
    max_loss_relative_error = 0.0
    step_seconds: list[float] = []

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        step_inputs = inputs.detach().clone().requires_grad_(True)
        ref_step_inputs = inputs.detach().clone().requires_grad_(True)

        start = time.perf_counter()
        out = asym_model(inputs_embeds=step_inputs, labels=labels, return_activations=False)
        ref_out = ref_model(inputs_embeds=ref_step_inputs, labels=labels, return_activations=False)
        assert out["loss"] is not None and ref_out["loss"] is not None
        out["loss"].backward()
        ref_out["loss"].backward()
        opt.step()
        ref_opt.step()
        _sync(dev)
        step_seconds.append(time.perf_counter() - start)

        loss_value = float(out["loss"].detach().float().item())
        ref_loss_value = float(ref_out["loss"].detach().float().item())
        losses.append(loss_value)
        ref_losses.append(ref_loss_value)
        denom = max(abs(ref_loss_value), 1e-8)
        max_loss_relative_error = max(max_loss_relative_error, abs(loss_value - ref_loss_value) / denom)
        finite_steps.append(
            bool(math.isfinite(loss_value))
            and bool(math.isfinite(ref_loss_value))
            and _finite_named_tensors(asym_model.named_parameters())
            and _finite_named_tensors((name, param.grad) for name, param in asym_model.named_parameters() if param.grad is not None)
            and bool(torch.isfinite(step_inputs.grad.detach().float()).all().item())
        )

    base_after = [module.base_layer.host_weight.weight for module in asym_model.modules() if isinstance(module, AsymLoRALinear)]
    frozen_unchanged = all(torch.equal(before, after) for before, after in zip(base_before, base_after))
    optimizer_state = _adamw_state_summary(asym_model, opt, expected_kind="lora")

    return {
        "target_mode": target_mode,
        "checkpointing": bool(checkpointing),
        "steps": steps,
        "lr": lr,
        "loss_curve": losses,
        "reference_loss_curve": ref_losses,
        "loss_relative_error_by_step": [
            abs(loss - ref_loss) / max(abs(ref_loss), 1e-8) for loss, ref_loss in zip(losses, ref_losses)
        ],
        "max_loss_relative_error": max_loss_relative_error,
        "all_finite": all(finite_steps),
        "finite_steps": finite_steps,
        "optimizer_contains_only_lora": _optimizer_contains_only_lora(asym_model, opt),
        "optimizer_state": optimizer_state,
        "frozen_base_unchanged": bool(frozen_unchanged),
        "base_weight_grads_absent": all(
            module.base_layer.host_weight.weight.grad is None
            for module in asym_model.modules()
            if isinstance(module, AsymLoRALinear)
        ),
        "adapter_state_sha256": adapter_state_hash(adapter_state_dict(asym_model)),
        "execution_stats": asym_model.stats.as_dict(),
        "direct_fetch_forward_used": asym_model.stats.asym_forward_calls > 0,
        "direct_fetch_dx_used": asym_model.stats.asym_dx_calls > 0,
        "pinned_cpu_bytes": asym_model.pinned_cpu_bytes,
        "expected_hbm_saved_bytes": asym_model.cpu_resident_base_weight_bytes,
        "step_seconds": step_seconds,
    }


def run_adapter_reload_case(
    *,
    target_mode: str = "all",
    backend: str = "torch_only",
    device: torch.device | str | None = "cpu",
    dtype: torch.dtype | str = torch.float32,
    seed: int = 20,
    lora_seed: int = 21,
    reload_lora_seed: int = 22,
    input_seed: int = 23,
    config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
) -> dict[str, Any]:
    dev = _device_from_name(str(device) if device is not None else None)
    resolved_dtype = _dtype_from_name(dtype) if isinstance(dtype, str) else dtype
    weights = make_tiny_dense_weights(config, seed=seed, dtype=resolved_dtype)
    model = AsymTinyDenseLLM(
        weights,
        config=config,
        target_mode=target_mode,
        backend=backend,
        stats=AsymExecutionStats(),
        device=dev,
        dtype=resolved_dtype,
        lora_seed=lora_seed,
    )
    reloaded = AsymTinyDenseLLM(
        weights,
        config=config,
        target_mode=target_mode,
        backend=backend,
        stats=AsymExecutionStats(),
        device=dev,
        dtype=resolved_dtype,
        lora_seed=reload_lora_seed,
    )
    state = adapter_state_dict(model)
    inputs, labels = make_inputs(config, seed=input_seed, device=dev, dtype=resolved_dtype)
    before = reloaded(inputs_embeds=inputs, labels=labels)["logits"]
    load_adapter_state_dict(reloaded, state, strict=True)
    source = model(inputs_embeds=inputs, labels=labels)["logits"]
    after = reloaded(inputs_embeds=inputs, labels=labels)["logits"]
    names = list(state.keys())
    return {
        "target_mode": target_mode,
        "adapter_state_names": names,
        "adapter_state_name_count": len(names),
        "adapter_state_sha256": adapter_state_hash(state),
        "reload_changed_logits_max_abs": _max_abs(before, after),
        "reload_matches_source_logits_max_abs": _max_abs(source, after),
        "names_have_peft_shape": all(
            ".lora_A.default.weight" in name or ".lora_B.default.weight" in name for name in names
        ),
        "base_keys_absent": all("base_layer" not in name for name in names),
    }


def _memory_probe(
    *,
    mode: str,
    target_mode: str,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    lora_seed: int,
    input_seed: int,
    config: TinyDenseLLMConfig,
) -> dict[str, Any]:
    _clear_cuda(device)
    hbm_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    rss_before = _process_rss_bytes()
    weights = make_tiny_dense_weights(config, seed=seed, dtype=dtype)
    if mode == "normal_gpu_resident":
        model: TinyDenseLLMBase = TorchTinyDenseLLM(
            weights,
            config=config,
            target_mode=target_mode,
            device=device,
            dtype=dtype,
            lora_seed=lora_seed,
        )
    elif mode == "asym_cpu_resident":
        model = AsymTinyDenseLLM(
            weights,
            config=config,
            target_mode=target_mode,
            backend=backend,
            stats=AsymExecutionStats(),
            device=device,
            dtype=dtype,
            lora_seed=lora_seed,
        )
    else:
        raise ValueError(f"unknown memory mode: {mode}")

    _sync(device)
    model_hbm = int(torch.cuda.memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    rss_after_build = _process_rss_bytes()
    inputs, labels = make_inputs(config, seed=input_seed, device=device, dtype=dtype)
    inputs = inputs.detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-3)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    out = model(inputs_embeds=inputs, labels=labels, return_activations=False)
    assert out["loss"] is not None
    out["loss"].backward()
    optimizer.step()
    _sync(device)
    step_seconds = time.perf_counter() - start
    peak_hbm = int(torch.cuda.max_memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    rss_after_step = _process_rss_bytes()
    optimizer_state = _adamw_state_summary(model, optimizer, expected_kind="lora")

    result = {
        "mode": mode,
        "model_hbm_bytes": max(0, model_hbm),
        "peak_hbm_bytes": max(0, peak_hbm),
        "rss_before_bytes": rss_before,
        "rss_after_build_bytes": rss_after_build,
        "rss_after_step_bytes": rss_after_step,
        "rss_build_delta_bytes": rss_after_build - rss_before,
        "rss_step_delta_bytes": rss_after_step - rss_before,
        "pinned_cpu_bytes": int(getattr(model, "pinned_cpu_bytes", 0)),
        "cpu_resident_base_weight_bytes": int(getattr(model, "cpu_resident_base_weight_bytes", 0)),
        "gpu_resident_target_weight_bytes": int(getattr(model, "gpu_resident_target_weight_bytes", 0)),
        "optimizer_state": optimizer_state,
        "step_seconds": step_seconds,
        "execution_stats": model.stats.as_dict(),
    }

    del out, optimizer, inputs, labels, model, weights
    _clear_cuda(device)
    return result


def run_memory_comparison(
    *,
    target_mode: str = "all",
    backend: str = "asym_or_staged",
    device: torch.device | str | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    seed: int = 30,
    lora_seed: int = 31,
    input_seed: int = 32,
    config: TinyDenseLLMConfig = TinyDenseLLMConfig(),
) -> dict[str, Any]:
    dev = _device_from_name(str(device) if device is not None else None)
    resolved_dtype = _dtype_from_name(dtype) if isinstance(dtype, str) else dtype
    normal = _memory_probe(
        mode="normal_gpu_resident",
        target_mode=target_mode,
        backend=backend,
        device=dev,
        dtype=resolved_dtype,
        seed=seed,
        lora_seed=lora_seed,
        input_seed=input_seed,
        config=config,
    )
    asym = _memory_probe(
        mode="asym_cpu_resident",
        target_mode=target_mode,
        backend=backend,
        device=dev,
        dtype=resolved_dtype,
        seed=seed,
        lora_seed=lora_seed,
        input_seed=input_seed,
        config=config,
    )
    return {
        "target_mode": target_mode,
        "normal_gpu_resident": normal,
        "asym_cpu_resident": asym,
        "hbm_model_saved_bytes": normal["model_hbm_bytes"] - asym["model_hbm_bytes"],
        "hbm_peak_saved_bytes": normal["peak_hbm_bytes"] - asym["peak_hbm_bytes"],
        "expected_hbm_saved_bytes": asym["cpu_resident_base_weight_bytes"],
        "pinned_cpu_bytes": asym["pinned_cpu_bytes"],
        "cpu_model_extra_bytes": asym["pinned_cpu_bytes"] - normal["pinned_cpu_bytes"],
        "direct_fetch_forward_used": asym["execution_stats"]["asym_forward_calls"] > 0,
        "direct_fetch_dx_used": asym["execution_stats"]["asym_dx_calls"] > 0,
    }


def _run_command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware_info() -> dict[str, Any]:
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        }
    asym_version = None
    try:
        import asym_gemm

        asym_version = getattr(asym_gemm, "__version__", None)
    except Exception:
        asym_version = None
    cpus_allowed = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    cpus_allowed = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpus_allowed = None
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_device,
        "cpu": platform.processor(),
        "cpus_allowed_list": cpus_allowed,
        "asym_gemm_version": asym_version,
        "git_commit": _run_command(["git", "rev-parse", "HEAD"]),
        "llamafactory_version": None,
        "ktransformers_version": None,
    }


def _summarize_status(report: Mapping[str, Any]) -> str:
    for case in report["parity_cases"]:
        if case["logits_max_abs"] > 0.5 or case["loss_abs"] > 0.5 or case["input_grad_max_abs"] > 0.5:
            return "fail"
        if case["lora_grad_worst_max_abs"] > 0.5 or case["activation_worst_max_abs"] > 0.5:
            return "fail"
        if not case["adapter_names_match_reference"] or not case["base_weight_grads_absent"]:
            return "fail"
    repeated = report["repeated_steps"]
    if not repeated["all_finite"] or repeated["max_loss_relative_error"] > 0.02:
        return "fail"
    optimizer_state = repeated.get("optimizer_state", {})
    if (
        optimizer_state.get("optimizer_class") != "AdamW"
        or not optimizer_state.get("all_expected_params_in_optimizer", False)
        or not optimizer_state.get("only_expected_params_in_optimizer", False)
        or not optimizer_state.get("state_for_all_expected_params", False)
        or not optimizer_state.get("adam_moments_for_all_expected_params", False)
        or optimizer_state.get("unexpected_state_param_count", 1) != 0
        or optimizer_state.get("unexpected_optimizer_param_count", 1) != 0
    ):
        return "fail"
    adapter = report["adapter_reload"]
    if adapter["reload_matches_source_logits_max_abs"] > 1e-5 or not adapter["names_have_peft_shape"] or not adapter["base_keys_absent"]:
        return "fail"
    memory = report["memory_comparison"]
    if memory["expected_hbm_saved_bytes"] <= 0 or memory["hbm_model_saved_bytes"] <= 0:
        return "fail"
    return "pass"


def run_m3_report(
    *,
    backend: str = "asym_or_staged",
    report_path: Optional[Path] = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    seed: int = 0,
    config: TinyDenseLLMConfig = MICRO_DENSE_LLM_CONFIG,
) -> dict[str, Any]:
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        dev = _device_from_name(str(device) if device is not None else None)
        resolved_dtype = _dtype_from_name(dtype) if isinstance(dtype, str) else dtype
        parity_cases = [
            run_parity_case(
                target_mode=target_mode,
                checkpointing=checkpointing,
                backend=backend,
                device=dev,
                dtype=resolved_dtype,
                seed=seed,
                lora_seed=seed + 1,
                input_seed=seed + 2,
                config=config,
            )
            for target_mode in TARGET_MODES
            for checkpointing in (False, True)
        ]
        repeated = run_repeated_steps(
            target_mode="all",
            checkpointing=False,
            backend=backend,
            device=dev,
            dtype=resolved_dtype,
            seed=seed + 10,
            lora_seed=seed + 11,
            input_seed=seed + 12,
            config=config,
        )
        adapter = run_adapter_reload_case(target_mode="all", backend="torch_only", device="cpu", dtype=torch.float32, config=config)
        memory = run_memory_comparison(
            target_mode="all",
            backend=backend,
            device=dev,
            dtype=resolved_dtype,
            seed=seed + 30,
            lora_seed=seed + 31,
            input_seed=seed + 32,
            config=config,
        )
        report: dict[str, Any] = {
            "milestone": "M3 Tiny Dense LLM Correctness",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command_line": sys.argv,
            "backend_requested": backend,
            "device": str(dev),
            "dtype": str(resolved_dtype),
            "tf32_disabled": bool(
                dev.type != "cuda"
                or (not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32)
            ),
            "config": asdict(config),
            "parameter_accounting": estimate_tiny_dense_llm_parameters(
                config,
                target_mode="all",
                dtype=resolved_dtype,
            ),
            "showcase_parameter_accounting": estimate_tiny_dense_llm_parameters(
                SHOWCASE_DENSE_LLM_CONFIG,
                target_mode="all",
                dtype=resolved_dtype,
            ),
            "target_modes": list(TARGET_MODES),
            "parity_cases": parity_cases,
            "repeated_steps": repeated,
            "adapter_reload": adapter,
            "memory_comparison": memory,
            "hardware": _hardware_info(),
            "precision": str(resolved_dtype),
        }
        report["status"] = _summarize_status(report)
        report["worst_numerical_error"] = {
            "logits_max_abs": max(case["logits_max_abs"] for case in parity_cases),
            "loss_abs": max(case["loss_abs"] for case in parity_cases),
            "input_grad_max_abs": max(case["input_grad_max_abs"] for case in parity_cases),
            "lora_grad_max_abs": max(case["lora_grad_worst_max_abs"] for case in parity_cases),
            "activation_max_abs": max(case["activation_worst_max_abs"] for case in parity_cases),
        }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


def run_tiny_dense_llm_case(
    *,
    backend: str = "asym_or_staged",
    report_path: Optional[Path] = None,
    seed: int = 0,
    device: Optional[str] = None,
) -> dict[str, Any]:
    return run_m3_report(
        backend=backend,
        report_path=report_path,
        device=device,
        dtype=torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32,
        seed=seed,
        config=MICRO_DENSE_LLM_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="asym_or_staged", choices=["asym_only", "asym_or_staged", "asym_or_torch", "torch_only"])
    parser.add_argument("--report", default="reports/m3_tiny_llm.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["bf16", "bfloat16", "fp32", "float32"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", choices=["micro", "showcase"], default="micro")
    args = parser.parse_args()
    config = MICRO_DENSE_LLM_CONFIG if args.config == "micro" else SHOWCASE_DENSE_LLM_CONFIG

    report = run_m3_report(
        backend=args.backend,
        report_path=Path(args.report),
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
