from __future__ import annotations

import torch
from torch import nn

from asym_gemm.training.lora import AsymLoRALinear
from asym_gemm.training.weight_offload import LoRAWeightOffloadCoordinator, install_lora_weight_offload


def test_generic_asym_lora_weight_offload_releases_forward_and_regathers_backward() -> None:
    torch.manual_seed(123)
    source = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    module = AsymLoRALinear(
        source,
        rank=4,
        alpha=8.0,
        backend="torch",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
    )
    module.train()
    model = nn.Sequential(module)
    coordinator = LoRAWeightOffloadCoordinator(pin_memory=False, persistence_threshold_numel=0)

    installed = install_lora_weight_offload(model, coordinator)

    assert installed == 2
    assert coordinator.summary()["weight_offload_group_count"] == 1
    assert module.lora_a.numel() == 0
    assert module.lora_b.numel() == 0

    x = torch.randn(3, 16, dtype=torch.bfloat16, requires_grad=True)
    loss = model(x).float().square().mean()

    assert module.lora_a.numel() == 0
    assert module.lora_b.numel() == 0

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()
    assert module.lora_a.grad is not None
    assert module.lora_b.grad is not None
    assert tuple(module.lora_a.grad.shape) == (4, 16)
    assert tuple(module.lora_b.grad.shape) == (8, 4)
