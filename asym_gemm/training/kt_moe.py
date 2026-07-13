from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .host_weight import tensor_nbytes
from .profile_ranges import is_profile_enabled, prof_range


DEFAULT_KT_METHOD = "AMXBF16_SFT"
KT_SFT_METHODS = (DEFAULT_KT_METHOD, "AMXINT8_SFT", "AMXINT4_SFT")


class KTBackendUnavailable(RuntimeError):
    """Raised when kt-kernel SFT support is not importable in this environment."""


def normalize_kt_method(method: str | None) -> str:
    normalized = str(method or DEFAULT_KT_METHOD).upper()
    if normalized not in KT_SFT_METHODS:
        raise ValueError(f"unsupported KT SFT method={method!r}; expected one of {KT_SFT_METHODS}")
    return normalized


def _load_kt_kernel_from_source_tree(root: Path) -> Any:
    init_py = root / "python" / "__init__.py"
    if not init_py.exists():
        raise ImportError(f"{init_py} does not exist")
    old_module = sys.modules.get("kt_kernel")
    spec = importlib.util.spec_from_file_location(
        "kt_kernel",
        init_py,
        submodule_search_locations=[str(init_py.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kt_kernel"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if old_module is None:
            sys.modules.pop("kt_kernel", None)
        else:
            sys.modules["kt_kernel"] = old_module
        raise
    return module


def _candidate_kt_kernel_roots() -> list[Path]:
    roots: list[Path] = []
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "ktransformers" / "kt-kernel"
        if candidate.exists():
            roots.append(candidate)
    return list(dict.fromkeys(roots))


def _import_kt_moe_wrapper() -> Any:
    errors: list[str] = []
    try:
        experts = importlib.import_module("kt_kernel.experts")
        return experts.KTMoEWrapper
    except Exception as exc:
        errors.append(f"installed/importable kt_kernel: {type(exc).__name__}: {exc}")

    for root in _candidate_kt_kernel_roots():
        try:
            _load_kt_kernel_from_source_tree(root)
            experts = importlib.import_module("kt_kernel.experts")
            return experts.KTMoEWrapper
        except Exception as exc:
            errors.append(f"{root}: {type(exc).__name__}: {exc}")

    raise KTBackendUnavailable(
        "backend=kt requires kt-kernel with SFT AMX support. "
        "Install/build ktransformers/kt-kernel with AMX SFT enabled. "
        f"Import/initialization errors: {' | '.join(errors)}"
    )


def _state_tensor(state: Mapping[str, Any], name: str) -> torch.Tensor:
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state field {name!r} is not a tensor")
    return value


def _stack_expert_weight(expert_states: Sequence[Mapping[str, Any]], name: str) -> torch.Tensor:
    if not expert_states:
        raise ValueError("cannot build KT expert weights from an empty expert list")
    return torch.stack([_state_tensor(expert_state, name) for expert_state in expert_states], dim=0).contiguous()


def _stack_expert_lora(expert_states: Sequence[Mapping[str, Any]], name: str, dtype: torch.dtype) -> torch.Tensor:
    if not expert_states:
        raise ValueError("cannot build KT LoRA buffers from an empty expert list")
    return torch.stack([_state_tensor(expert_state, name).to(dtype=dtype) for expert_state in expert_states], dim=0).contiguous()


def _normalize_kt_lora_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if dtype is torch.bfloat16:
        return torch.bfloat16
    normalized = str(dtype).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    raise ValueError("backend=kt currently supports BF16 LoRA buffers only")


class _KTRoutedMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        module: "KTRoutedExpertMoE",
        gate_lora_a: torch.Tensor,
        gate_lora_b: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(f"KT routed MoE expects hidden_states [tokens, hidden], got {tuple(hidden_states.shape)}")
        qlen = int(hidden_states.shape[0])
        hidden = int(hidden_states.shape[1])
        if hidden != module.hidden_size:
            raise ValueError(f"KT routed MoE expected hidden size {module.hidden_size}, got {hidden}")

        ctx.module = module
        ctx.qlen = qlen
        ctx.hidden_size = hidden
        ctx.hidden_dtype = hidden_states.dtype
        ctx.hidden_device = hidden_states.device
        ctx.weights_shape = tuple(topk_weights.shape)
        ctx.weights_dtype = topk_weights.dtype
        ctx.profile_enabled = is_profile_enabled()

        with prof_range("forward.kt_moe.submit_copy_in"):
            module.wrapper.submit_forward(
                hidden_states.detach().contiguous(),
                topk_ids.detach().contiguous(),
                topk_weights.detach().contiguous(),
                save_for_backward=True,
            )
        with prof_range("forward.kt_moe.sync_copy_out"):
            output = module.wrapper.sync_forward(output_device=hidden_states.device)
        return output.view(qlen, hidden).to(dtype=hidden_states.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        module: KTRoutedExpertMoE = ctx.module
        module.zero_lora_grad_buffers()
        with prof_range("backward.kt_moe.backward_sft", enabled=ctx.profile_enabled):
            backward_out = module.wrapper.backward(
                grad_output.reshape(ctx.qlen, ctx.hidden_size).contiguous(),
                output_device=ctx.hidden_device,
            )
        if isinstance(backward_out, tuple) and len(backward_out) == 2:
            grad_input, grad_weights = backward_out
        elif isinstance(backward_out, tuple) and len(backward_out) == 3:
            grad_input, _, grad_weights = backward_out
        else:
            raise RuntimeError("KTMoEWrapper.backward returned an unexpected format")
        if module.stats is not None and hasattr(module.stats, "kt_backward_calls"):
            module.stats.kt_backward_calls += 1

        grad_input = grad_input.view(ctx.qlen, ctx.hidden_size).to(device=ctx.hidden_device, dtype=ctx.hidden_dtype)
        grad_weights = grad_weights.view(ctx.weights_shape).to(device=ctx.hidden_device, dtype=ctx.weights_dtype)
        lora_grads = module.lora_grad_tensors()
        return (
            grad_input,
            None,
            grad_weights,
            None,
            *lora_grads,
        )


class KTRoutedExpertMoE(nn.Module):
    """Local one-GPU KT SFT routed expert adapter for the MoE benchmark."""

    def __init__(
        self,
        expert_states: Sequence[Mapping[str, Any]],
        *,
        config: Any,
        device: torch.device,
        layer_idx: int,
        method: str = "AMXBF16_SFT",
        cpuinfer_threads: int | None = None,
        threadpool_count: int = 1,
        chunked_prefill_size: int | None = None,
        max_cache_depth: int = 1,
        lora_dtype: torch.dtype | str = torch.bfloat16,
        stats: Any | None = None,
    ) -> None:
        super().__init__()
        method = normalize_kt_method(method)
        if method != DEFAULT_KT_METHOD:
            raise NotImplementedError(
                f"{method} is exposed as a planned KT SFT method, but the local tensor-loading path "
                "is implemented for AMXBF16_SFT first. Add KT quantized weight loading before using INT methods."
            )
        self.method = method
        self.device_hint = torch.device(device)
        self.layer_idx = int(layer_idx)
        self.num_experts = int(config.num_experts)
        self.num_experts_per_tok = int(config.top_k)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        self.lora_rank = int(config.lora_rank)
        self.lora_alpha = float(config.lora_alpha)
        self.stats = stats
        self._lora_dtype = _normalize_kt_lora_dtype(lora_dtype)

        KTMoEWrapper = _import_kt_moe_wrapper()
        threads = int(cpuinfer_threads or max(1, os.cpu_count() or 1))
        pools = int(threadpool_count)
        cache_depth = int(max_cache_depth)
        if threads <= 0:
            raise ValueError("kt_cpu_threads must be positive")
        if pools <= 0:
            raise ValueError("kt_threadpool_count must be positive")
        if cache_depth <= 0:
            raise ValueError("kt_max_cache_depth must be positive")
        chunk = int(chunked_prefill_size or max(1, int(config.logical_tokens), int(config.batch_size) * int(config.seq_len)))
        self.wrapper = KTMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=threads,
            threadpool_count=pools,
            weight_path="",
            chunked_prefill_size=chunk,
            method=self.method,
            mode="sft",
            num_gpu_experts=0,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            max_cache_depth=cache_depth,
        )

        gate_weight = _stack_expert_weight(expert_states, "gate_weight").to(device="cpu", dtype=torch.bfloat16)
        up_weight = _stack_expert_weight(expert_states, "up_weight").to(device="cpu", dtype=torch.bfloat16)
        down_weight = _stack_expert_weight(expert_states, "down_weight").to(device="cpu", dtype=torch.bfloat16)
        self._frozen_weight_bytes = tensor_nbytes(gate_weight) + tensor_nbytes(up_weight) + tensor_nbytes(down_weight)
        physical_to_logical = torch.arange(self.num_experts, dtype=torch.int64, device="cpu")
        self.wrapper.load_weights_from_tensors(gate_weight, up_weight, down_weight, physical_to_logical)

        self.gate_lora_a = nn.Parameter(_stack_expert_lora(expert_states, "gate_lora_a", self._lora_dtype))
        self.gate_lora_b = nn.Parameter(_stack_expert_lora(expert_states, "gate_lora_b", self._lora_dtype))
        self.up_lora_a = nn.Parameter(_stack_expert_lora(expert_states, "up_lora_a", self._lora_dtype))
        self.up_lora_b = nn.Parameter(_stack_expert_lora(expert_states, "up_lora_b", self._lora_dtype))
        self.down_lora_a = nn.Parameter(_stack_expert_lora(expert_states, "down_lora_a", self._lora_dtype))
        self.down_lora_b = nn.Parameter(_stack_expert_lora(expert_states, "down_lora_b", self._lora_dtype))

        self.register_buffer("_grad_gate_lora_a", torch.zeros_like(self.gate_lora_a, device="cpu"))
        self.register_buffer("_grad_gate_lora_b", torch.zeros_like(self.gate_lora_b, device="cpu"))
        self.register_buffer("_grad_up_lora_a", torch.zeros_like(self.up_lora_a, device="cpu"))
        self.register_buffer("_grad_up_lora_b", torch.zeros_like(self.up_lora_b, device="cpu"))
        self.register_buffer("_grad_down_lora_a", torch.zeros_like(self.down_lora_a, device="cpu"))
        self.register_buffer("_grad_down_lora_b", torch.zeros_like(self.down_lora_b, device="cpu"))

        self.wrapper.init_lora_weights(
            self.gate_lora_a,
            self.gate_lora_b,
            self.up_lora_a,
            self.up_lora_b,
            self.down_lora_a,
            self.down_lora_b,
            self._grad_gate_lora_a,
            self._grad_gate_lora_b,
            self._grad_up_lora_a,
            self._grad_up_lora_b,
            self._grad_down_lora_a,
            self._grad_down_lora_b,
        )

    @property
    def frozen_weight_bytes(self) -> int:
        return int(self._frozen_weight_bytes)

    @property
    def pinned_cpu_bytes(self) -> int:
        return 0

    @property
    def kt_cpu_weight_bytes(self) -> int:
        return int(self._frozen_weight_bytes)

    @property
    def kt_lora_bytes(self) -> int:
        return sum(tensor_nbytes(param) for param in self.lora_parameters())

    @property
    def kt_lora_grad_bytes(self) -> int:
        return sum(tensor_nbytes(grad) for grad in self.lora_grad_tensors())

    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.gate_lora_a,
            self.gate_lora_b,
            self.up_lora_a,
            self.up_lora_b,
            self.down_lora_a,
            self.down_lora_b,
        )

    def lora_grad_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self._grad_gate_lora_a,
            self._grad_gate_lora_b,
            self._grad_up_lora_a,
            self._grad_up_lora_b,
            self._grad_down_lora_a,
            self._grad_down_lora_b,
        )

    def zero_lora_grad_buffers(self) -> None:
        for grad in self.lora_grad_tensors():
            grad.zero_()

    def update_lora_weights(self) -> None:
        with prof_range("optimizer.kt_lora_update"):
            self.wrapper.update_lora_weights()
        if self.stats is not None and hasattr(self.stats, "kt_lora_update_calls"):
            self.stats.kt_lora_update_calls += 1

    def forward(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        if hidden_states.device != self.device_hint:
            hidden_states = hidden_states.to(device=self.device_hint)
        if topk_ids.shape != topk_weights.shape:
            raise ValueError(f"topk_ids/topk_weights shape mismatch: {tuple(topk_ids.shape)} vs {tuple(topk_weights.shape)}")
        if topk_ids.shape[1] != self.num_experts_per_tok:
            raise ValueError(f"KT expected top_k={self.num_experts_per_tok}, got {topk_ids.shape[1]}")
        if self.stats is not None and hasattr(self.stats, "kt_forward_calls"):
            self.stats.kt_forward_calls += 1
        return _KTRoutedMoEFunction.apply(
            hidden_states,
            topk_ids.to(dtype=torch.long),
            topk_weights.to(dtype=torch.float32),
            self,
            self.gate_lora_a,
            self.gate_lora_b,
            self.up_lora_a,
            self.up_lora_b,
            self.down_lora_a,
            self.down_lora_b,
        )
