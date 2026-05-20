from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, VALID_ASYM_PRECISIONS


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
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        precision = str(precision).lower()
        if precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        self.base = AsymFrozenLinear(
            weight,
            backend=backend,
            pin_memory=device.type == "cuda",
            stats=stats,
            precision=precision,
        )
        self.lora_a = nn.Parameter(torch.randn(rank, weight.shape[1], device=device, dtype=torch.float32) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(weight.shape[0], rank, device=device, dtype=torch.float32) * 0.01)
        self.scaling = alpha / rank
        self.precision = precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        lora = (x.float() @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
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
    ) -> None:
        super().__init__()
        self.register_buffer("base_weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        self.lora_a = nn.Parameter(torch.randn(rank, weight.shape[1], device=device, dtype=torch.float32) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(weight.shape[0], rank, device=device, dtype=torch.float32) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.base_weight.t()
        lora = (x.float() @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
        return base + lora.to(dtype=base.dtype)


class AsymMLP(nn.Module):
    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        rank: int,
        alpha: float,
        backend: str,
        stats: AsymExecutionStats,
        device: torch.device,
        dtype: torch.dtype,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        precision = str(precision).lower()
        self.fc1 = AsymLoRALinear(
            w1,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            dtype=dtype,
            precision=precision,
        )
        self.fc2 = AsymLoRALinear(
            w2,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            dtype=dtype,
            precision=precision,
        )
        self.precision = precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.fc1.base.pinned_cpu_bytes + self.fc2.base.pinned_cpu_bytes

    @property
    def expected_hbm_saved_bytes(self) -> int:
        return self.fc1.base.weight_hbm_saved_bytes + self.fc2.base.weight_hbm_saved_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.expected_hbm_saved_bytes


class TorchMLP(nn.Module):
    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        rank: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.fc1 = TorchLoRALinear(w1, rank=rank, alpha=alpha, device=device, dtype=dtype)
        self.fc2 = TorchLoRALinear(w2, rank=rank, alpha=alpha, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def copy_lora(src: AsymMLP, dst: TorchMLP) -> None:
    with torch.no_grad():
        dst.fc1.lora_a.copy_(src.fc1.lora_a)
        dst.fc1.lora_b.copy_(src.fc1.lora_b)
        dst.fc2.lora_a.copy_(src.fc2.lora_a)
        dst.fc2.lora_b.copy_(src.fc2.lora_b)


def lora_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [param for name, param in model.named_parameters() if "lora_" in name]


def optimizer_contains_only(params: Iterable[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> bool:
    expected = {id(param) for param in params}
    actual = {id(param) for group in optimizer.param_groups for param in group["params"]}
    return actual == expected


__all__ = [
    "AsymLoRALinear",
    "AsymMLP",
    "TorchLoRALinear",
    "TorchMLP",
    "copy_lora",
    "lora_parameters",
    "optimizer_contains_only",
]
