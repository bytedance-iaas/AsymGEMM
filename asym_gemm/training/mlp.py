from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import AsymExecutionStats
from .lora import AsymLoRALinear, TorchLoRALinear, copy_lora, lora_parameters


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
        lora_dtype: torch.dtype | str = torch.bfloat16,
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
            lora_dtype=lora_dtype,
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
            lora_dtype=lora_dtype,
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
        lora_dtype: torch.dtype | str = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.fc1 = TorchLoRALinear(w1, rank=rank, alpha=alpha, device=device, dtype=dtype, lora_dtype=lora_dtype)
        self.fc2 = TorchLoRALinear(w2, rank=rank, alpha=alpha, device=device, dtype=dtype, lora_dtype=lora_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


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
