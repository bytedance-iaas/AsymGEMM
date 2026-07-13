"""Surgical activation offload for a DENSE MLP (Qwen3MLP / LlamaMLP shape).

A dense MLP is the `exp` of a dense LLM: this reuses the Qwen3 expert engine
(`AsymQwen3Experts`) as a single expert (E=1, all tokens routed to expert 0) so
`ASYMM_EXPERT_ACT_OFFLOAD` offloads the `silu(gate)*up` intermediate to CPU and
runs the down-proj LoRA backward on CPU through AsymGEMM — never staging the
intermediate back to GPU. No new kernels and no new autograd function; the CPU
LoRA grads are bit-identical to the on-GPU path (validated, see fix_dense_v2.md).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import AsymExecutionStats
from .qwen3_moe import AsymQwen3Experts


_DENSE_MLP_LEAVES = ("gate_proj", "up_proj", "down_proj")


class _DenseMLPExpertSource(nn.Module):
    """Adapter that presents a dense `Qwen3MLP` as a packed E=1 expert module.

    `AsymQwen3Experts` expects `gate_up_proj` [E, 2I, H], `down_proj` [E, H, I],
    integer `num_experts/hidden_dim/intermediate_dim`, a callable `act_fn`, and a
    routing forward signature — exactly what `is_qwen3_experts` checks. We fuse
    gate+up into one expert and keep the *same* base tensors (CPU-first storage).
    """

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        gate_w = mlp.gate_proj.weight.detach()  # [I, H]
        up_w = mlp.up_proj.weight.detach()       # [I, H]
        down_w = mlp.down_proj.weight.detach()   # [H, I]
        self.gate_up_proj = nn.Parameter(
            torch.cat([gate_w, up_w], dim=0).unsqueeze(0).contiguous(), requires_grad=False
        )  # [1, 2I, H]
        self.down_proj = nn.Parameter(down_w.unsqueeze(0).contiguous(), requires_grad=False)  # [1, H, I]
        self.num_experts = 1
        self.hidden_dim = int(gate_w.shape[1])
        self.intermediate_dim = int(gate_w.shape[0])
        self.act_fn = getattr(mlp, "act_fn", F.silu)
        self.config = getattr(mlp, "config", None)

    def forward(self, hidden_states, top_k_index, top_k_weights):  # never called; satisfies the matcher
        return hidden_states


def build_dense_mlp_expert_engine(
    mlp: nn.Module,
    *,
    backend: str,
    precision: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> AsymQwen3Experts:
    return AsymQwen3Experts(
        _DenseMLPExpertSource(mlp),
        backend=backend,
        precision=precision,
        offload=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        stats=stats,
        strict=strict,
    )


def is_dense_mlp_module(module: nn.Module) -> bool:
    """A plain dense MLP: has gate/up/down nn.Linear leaves and is NOT a MoE block."""
    from .qwen3_moe import is_qwen3_experts, is_qwen3_moe_block
    from .qwen35_moe import is_qwen35_moe_block
    from .llama4_moe import is_llama4_moe

    if is_qwen3_experts(module) or is_qwen3_moe_block(module) or is_qwen35_moe_block(module) or is_llama4_moe(module):
        return False
    return all(isinstance(getattr(module, leaf, None), nn.Linear) for leaf in _DENSE_MLP_LEAVES)


class AsymDenseMLP(nn.Module):
    """Drop-in replacement for a dense MLP that offloads its activation via the expert engine."""

    _is_asym_dense_mlp = True

    def __init__(self, engine: AsymQwen3Experts) -> None:
        super().__init__()
        self.engine = engine

    @property
    def cpu_resident_base_bytes(self) -> int:
        return int(getattr(self.engine, "cpu_resident_base_bytes", 0))

    @property
    def gpu_resident_base_bytes(self) -> int:
        return int(getattr(self.engine, "gpu_resident_base_bytes", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, hidden = x.shape
        flat = x.reshape(-1, hidden)
        m = flat.shape[0]
        # one expert, all tokens routed to expert 0 with weight 1 -> exactly the dense MLP
        top_k_index = torch.zeros(m, 1, dtype=torch.long, device=flat.device)
        top_k_weights = torch.ones(m, 1, dtype=flat.dtype, device=flat.device)
        out = self.engine(flat, top_k_index, top_k_weights)
        return out.reshape(*lead, hidden)


__all__ = [
    "AsymDenseMLP",
    "build_dense_mlp_expert_engine",
    "is_dense_mlp_module",
]
