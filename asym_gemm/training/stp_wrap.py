"""Stage I4 full-TP wrap (gb200_tp.md): two-branch decoder layers over the sharded arena.

Architecture (audit-resolved deviation from the doc's single-two-branch-Function plan,
recorded in the Decision Log): instead of restructuring the fine-grained Functions, each
decoder layer runs TWO INSTANCES of the EXISTING single-device machinery — branch0 on
dev0 over shard-0 HostWeights, branch1 on dev1 over shard-1 — connected ONLY by the I4
boundary Functions:

    x0, x1 = Bcast01Fn(x)                  # per-layer re-bcast; bwd drops g1 (g1==g0)
    h0,h1 = TPRegionFn(norm0(x0), norm1(x1))
    a_i   = self_attn_i(h_i)               # col shards -> half heads; row o -> PARTIAL
    p0,p1 = AllReduce2Fn(a0, a1);  y_i = x_i + p_i
    g0,g1 = TPRegionFn(pnorm0(y0), pnorm1(y1))
    q_i   = mlp_i(g_i)                     # fg MLP over shards; down -> PARTIAL
    r0,r1 = AllReduce2Fn(q0, q1);  out_i = y_i + r_i
    return Join01Fn(out0, out1)            # bwd hands the SAME grad to both branches

The per-layer Bcast/Join (instead of a residual sidecar) costs one extra [M,H] NVLink
copy each way per layer (~2 ms at s20000b8) and buys: gradient-checkpointing safety
(recompute recreates x1 from the saved x0 — exact, x0==x1 bit-identical) and zero
model-level state. g1-drop stays correct because every region boundary already summed
both branches' contributions into BOTH outputs, so g_x1 == g_x0 by construction.

LoRA layout falls out of constructing the wrappers over shard-shaped HostWeights:
  col (q/k/v/gate/up): B is auto-sized [N_i, r] per branch (sliced, both REGISTERED
      trainables); A [r, K] would duplicate -> branch1's A is DEMOTED to a plain-tensor
      MIRROR (not a Parameter). dA = dA0 + to0(dA1) merged post-backward; mirror.data
      resynced from the owner after optimizer.step.
  row (o/down): A auto-sized [r, K_i] per branch (sliced, registered); B [N, r]
      replicated -> branch1's B demoted to a mirror, same merge/resync.
Logical trainable numel is IDENTICAL to |1 (full A + row-split B / full B + col-split A).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .stp_functions import AllReduce2Fn, Bcast01Fn, Join01Fn, TPRegionFn
from .stp_layout import split_points
from .stp_runtime import get_runtime

try:  # transformers 5.x — REQUIRED so gradient_checkpointing_enable() reaches our layer
    from transformers.modeling_layers import GradientCheckpointingLayer as _GCBase
except ImportError:  # pragma: no cover - older transformers
    _GCBase = nn.Module

__all__ = [
    "build_stp_full_tp",
    "finalize_stp_placement",
    "stp_post_backward_merge",
    "stp_register_optimizer_resync",
]

_ADAPTER = "default"


def _shard_full_weight(name: str, weight: torch.Tensor, kind: str):
    """Return (s0, s1) contiguous pinned shards. col = zero-copy views of the original
    (which stays alive as the arena); row = one new pinned carrier split in halves, the
    original is freed when the wrapped module tree drops its references."""
    if kind == "col":
        n0 = split_points("col", weight.shape)
        return weight[:n0], weight[n0:]
    k0 = split_points("row", weight.shape)
    n, k = int(weight.shape[0]), int(weight.shape[1])
    carrier = torch.empty(n * k, dtype=weight.dtype).pin_memory()
    s0 = carrier[: n * k0].view(n, k0)
    s1 = carrier[n * k0 :].view(n, k - k0)
    s0.copy_(weight[:, :k0])
    s1.copy_(weight[:, k0:])
    if not torch.equal(torch.cat([s0, s1], dim=1), weight):
        raise RuntimeError(f"{name}: row shard reassembly mismatch")
    return s0, s1


def _demote_to_mirror(linear: nn.Linear, owner: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Replace a replicated LoRA piece's Parameter with a plain requires_grad tensor
    (invisible to named_parameters/optimizer) initialized from the owner param."""
    del linear._parameters["weight"]
    mirror = owner.detach().to(device).clone().requires_grad_(True)
    linear.weight = mirror
    return mirror


def _copy_param(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.to(dst.device, dst.dtype))


def _shared_frozen_norm(source_norm: nn.Module) -> nn.Module:
    """Branch1 norm = AsymFrozenRMSNorm sharing the SOURCE norm's pinned host weight
    (zero-copy dedup; weight stages to the consuming device per call)."""
    from .offload import AsymFrozenRMSNorm, adopt_host_weight

    if isinstance(source_norm, AsymFrozenRMSNorm):
        return AsymFrozenRMSNorm(source_norm, host_weight=source_norm.host_weight)
    weight = getattr(source_norm, "weight", None)
    if not torch.is_tensor(weight):
        raise RuntimeError(f"cannot build shared frozen norm from {type(source_norm).__name__}")
    host_weight = adopt_host_weight(
        "stp1_rms_norm", weight.detach().cpu(), "norms", require_2d=False, pin_memory_policy="none"
    )
    return AsymFrozenRMSNorm(source_norm, host_weight=host_weight)


def _norm_weight_eps(norm_module: nn.Module, default_eps: float) -> tuple[torch.Tensor, float]:
    """Extract (weight, eps) from an HF Qwen3RMSNorm OR the LF surgery's
    AsymFrozenRMSNorm (weight lives in a pinned HostWeight, eps under .eps)."""
    eps = float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", default_eps)))
    weight = getattr(norm_module, "weight", None)
    if not torch.is_tensor(weight):
        host_weight = getattr(norm_module, "host_weight", None)
        if host_weight is None:
            raise RuntimeError(f"cannot extract norm weight from {type(norm_module).__name__}")
        weight = host_weight.weight
    return weight, eps


class StpDecoderLayer(_GCBase):
    """Two-branch decoder layer. Child names preserve the |1 layout (self_attn / mlp /
    input_layernorm / post_attention_layernorm) so component classification and
    residency validation keep working; branch1 children carry the _stp1 suffix.

    Subclasses GradientCheckpointingLayer so gradient_checkpointing_enable() (incl. the
    unsloth save-on-cpu variant) checkpoints the WHOLE two-branch layer; recompute
    recreates x1 from the saved x0 via Bcast01Fn (exact — x0==x1 bit-identical) and
    replays the deterministic P2P exchanges."""

    def __init__(self, *, self_attn, mlp, input_layernorm, post_attention_layernorm,
                 self_attn_stp1, mlp_stp1, input_layernorm_stp1, post_attention_layernorm_stp1):
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn_stp1 = self_attn_stp1
        self.mlp_stp1 = mlp_stp1
        self.input_layernorm_stp1 = input_layernorm_stp1
        self.post_attention_layernorm_stp1 = post_attention_layernorm_stp1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        rt = get_runtime()
        dev1 = rt.d[1]
        if position_embeddings is None:
            raise RuntimeError("StpDecoderLayer requires position_embeddings from the model")
        cos, sin = position_embeddings
        # no-grad operand copies for branch1 (rope caches + mask); recompute-safe
        cos1 = rt.bcast01(cos.detach())
        sin1 = rt.bcast01(sin.detach())
        if torch.is_tensor(attention_mask):
            mask1 = rt.bcast01(attention_mask.detach())
        else:
            mask1 = attention_mask

        x0, x1 = Bcast01Fn.apply(hidden_states)

        n0 = self.input_layernorm(x0)
        with torch.cuda.device(dev1):
            n1 = self.input_layernorm_stp1(x1)
        h0, h1 = TPRegionFn.apply(n0, n1)
        a0, _ = self.self_attn(
            hidden_states=h0,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        )
        with torch.cuda.device(dev1):
            a1, _ = self.self_attn_stp1(
                hidden_states=h1,
                position_embeddings=(cos1, sin1),
                attention_mask=mask1,
                past_key_values=None,
                use_cache=False,
                **kwargs,
            )
        p0, p1 = AllReduce2Fn.apply(a0, a1)
        y0 = x0 + p0
        with torch.cuda.device(dev1):
            y1 = x1 + p1

        m0 = self.post_attention_layernorm(y0)
        with torch.cuda.device(dev1):
            m1 = self.post_attention_layernorm_stp1(y1)
        g0, g1 = TPRegionFn.apply(m0, m1)
        q0 = self.mlp(g0)
        with torch.cuda.device(dev1):
            q1 = self.mlp_stp1(g1)
        r0, r1 = AllReduce2Fn.apply(q0, q1)
        out0 = y0 + r0
        with torch.cuda.device(dev1):
            out1 = y1 + r1
        return Join01Fn.apply(out0, out1)


class _ShardSourceMLP(nn.Module):
    """nn.Linear-shaped source for build_finegrained_dense_mlp over shard tensors."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, act_fn) -> None:
        super().__init__()
        self.gate_proj = self._linear_over(gate)
        self.up_proj = self._linear_over(up)
        self.down_proj = self._linear_over(down)
        self.act_fn = act_fn

    @staticmethod
    def _linear_over(weight: torch.Tensor) -> nn.Linear:
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=weight.dtype)
        linear._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
        return linear


def build_stp_full_tp(
    model: nn.Module,
    *,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    backend: str,
    stats,
    strict: bool,
) -> dict[str, int]:
    """Convert every already-wrapped dense decoder layer into an StpDecoderLayer.

    Runs AFTER the standard |1 wrap (attention leaves + fg MLP exist with FULL-weight
    HostWeights and full LoRA params) and BEFORE freeze/validate/report. The original
    modules' LoRA values are copied into the branch pieces, then the originals are
    dropped (their full row weights free; col shard views keep the arena storage)."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm

    from .attention_activation_offload import (
        AsymActivationOffloadLoRALinear,
        AttentionActivationOffloadContext,
        install_attention_saved_tensor_offload,
    )
    from .dense_mlp_finegrained import AsymFinegrainedDenseMLP, build_finegrained_dense_mlp
    from .offload import adopt_host_weight

    rt = get_runtime()
    dev0, dev1 = rt.d[0], rt.d[1]
    mirror_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    param_map: dict[str, dict[str, Any]] = {}
    counts = {"layers": 0, "col": 0, "row": 0}

    decoder = model.model if hasattr(model, "model") else model
    layers = decoder.layers if hasattr(decoder, "layers") else decoder.model.layers
    config = model.config

    from .qwen3_moe import AsymQwen3MoeBlock

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp
        is_moe = isinstance(mlp, AsymQwen3MoeBlock)
        if not is_moe and not isinstance(mlp, AsymFinegrainedDenseMLP):
            raise RuntimeError(
                f"layer {layer_idx}: sTP full-TP requires the fine-grained dense MLP path "
                f"(recomp-off-full-fg) or an AsymQwen3MoeBlock (ASYM_STP_MOE), got {type(mlp).__name__}"
            )
        layer_tag = f"model.layers.{layer_idx}"

        # ---- attention branches ----
        ctx0 = AttentionActivationOffloadContext()
        ctx1 = AttentionActivationOffloadContext()
        with torch.device("meta"):
            if getattr(config, "model_type", "") == "qwen3_moe":
                from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

                attn1 = Qwen3MoeAttention(config=config, layer_idx=layer_idx)
            else:
                attn1 = Qwen3Attention(config=config, layer_idx=layer_idx)
        # branch1 q/k norms: AsymFrozenRMSNorm SHARING branch0's pinned host weight —
        # identical class/math as |1 (bit-identity across branches), stages to x.device
        # per call, and every |1 validator/mover handles it natively. Falls back to
        # wrapping a plain HF norm if the source was never frozen-wrapped.
        for norm_name in ("q_norm", "k_norm"):
            setattr(attn1, norm_name, _shared_frozen_norm(getattr(attn, norm_name)))

        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            original = getattr(attn, role)
            hw_full = original.base_layer.host_weight
            kind = "row" if role == "o_proj" else "col"
            s0, s1 = _shard_full_weight(f"{layer_tag}.self_attn.{role}", hw_full.weight, kind)
            hw0 = adopt_host_weight(f"{layer_tag}.self_attn.{role}.stp0", s0, "attention", pin_memory_policy="auto", strict=strict)
            hw1 = adopt_host_weight(f"{layer_tag}.self_attn.{role}.stp1", s1, "attention", pin_memory_policy="auto", strict=strict)
            common = dict(rank=lora_rank, alpha=lora_alpha, backend=backend, stats=stats,
                          precision="bf16", lora_dropout=lora_dropout, projection_role=role)
            branch0 = AsymActivationOffloadLoRALinear.from_host_weight(
                hw0, device=dev0, attention_context=(ctx0 if role != "o_proj" else None), **common)
            with torch.cuda.device(dev1):
                branch1 = AsymActivationOffloadLoRALinear.from_host_weight(
                    hw1, device=dev1, attention_context=(ctx1 if role != "o_proj" else None), **common)
            orig_a = original.lora_A[_ADAPTER].weight
            orig_b = original.lora_B[_ADAPTER].weight
            a0 = branch0.lora_A[_ADAPTER].weight
            b0 = branch0.lora_B[_ADAPTER].weight
            if kind == "col":
                n0 = int(s0.shape[0])
                _copy_param(a0, orig_a)
                _copy_param(b0, orig_b[:n0])
                _copy_param(branch1.lora_B[_ADAPTER].weight, orig_b[n0:])
                mirror = _demote_to_mirror(branch1.lora_A[_ADAPTER], a0, dev1)
                mirror_pairs.append((a0, mirror))
                counts["col"] += 1
            else:
                k0 = int(s0.shape[1])
                _copy_param(b0, orig_b)
                _copy_param(a0, orig_a[:, :k0])
                _copy_param(branch1.lora_A[_ADAPTER].weight, orig_a[:, k0:])
                mirror = _demote_to_mirror(branch1.lora_B[_ADAPTER], b0, dev1)
                mirror_pairs.append((b0, mirror))
                counts["row"] += 1
            param_map[f"{layer_tag}.self_attn.{role}"] = {"kind": kind, "split": int(s0.shape[0] if kind == "col" else s0.shape[1])}
            setattr(attn, role, branch0)
            setattr(attn1, role, branch1)

        # ---- MLP branches ----
        if is_moe:
            # sEP static EP-2 (gb200_ep.md E1 / I7): experts [0,E/2) on dev0, [E/2,E) on
            # dev1 over SHARED pinned banks (dim-0 views); replicated frozen routers give
            # identical routing; each branch scatters its PARTIAL and the existing
            # AllReduce2Fn at the mlp slot completes the sum. Expert LoRA slices are
            # DISJOINT per branch — no mirrors, both branches' params train directly.
            from .stp_moe import _LORA_NAMES, build_ep_branch_block

            num_experts = int(mlp.experts.num_experts)
            half = num_experts // 2
            with torch.cuda.device(dev0):
                mlp0 = build_ep_branch_block(mlp, 0, half, dev0)
            with torch.cuda.device(dev1):
                mlp1 = build_ep_branch_block(mlp, half, num_experts, dev1)
            mlp0.profile_prefix = f"{layer_tag}.mlp"
            mlp0.experts.profile_prefix = f"{layer_tag}.mlp.experts"
            mlp1.profile_prefix = f"{layer_tag}.mlp_stp1"
            mlp1.experts.profile_prefix = f"{layer_tag}.mlp_stp1.experts"
            for lora_name in _LORA_NAMES:
                param_map[f"{layer_tag}.mlp.experts.{lora_name}"] = {"kind": "moe_ep", "split": half}
            counts["moe"] = counts.get("moe", 0) + 1
        if not is_moe:
            shard_pairs = {}
            for proj_name, kind in (("gate_proj", "col"), ("up_proj", "col"), ("down_proj", "row")):
                proj = getattr(mlp, proj_name)
                shard_pairs[proj_name] = (
                    _shard_full_weight(f"{layer_tag}.mlp.{proj_name}", proj.base_layer.host_weight.weight, kind),
                    proj.lora_A[_ADAPTER].weight,
                    proj.lora_B[_ADAPTER].weight,
                )
                counts["col" if kind == "col" else "row"] += 1
            act_fn = mlp.activation_fn
            src0 = _ShardSourceMLP(shard_pairs["gate_proj"][0][0], shard_pairs["up_proj"][0][0], shard_pairs["down_proj"][0][0], act_fn)
            src1 = _ShardSourceMLP(shard_pairs["gate_proj"][0][1], shard_pairs["up_proj"][0][1], shard_pairs["down_proj"][0][1], act_fn)
            with torch.cuda.device(dev0):
                mlp0 = build_finegrained_dense_mlp(
                    src0, backend=backend, precision="bf16", lora_rank=lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, stats=stats, strict=strict, profile_prefix=f"{layer_tag}.mlp")
            with torch.cuda.device(dev1):
                mlp1 = build_finegrained_dense_mlp(
                    src1, backend=backend, precision="bf16", lora_rank=lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, stats=stats, strict=strict, profile_prefix=f"{layer_tag}.mlp_stp1")
            for proj_name, kind in (("gate_proj", "col"), ("up_proj", "col"), ("down_proj", "row")):
                orig_a = shard_pairs[proj_name][1]
                orig_b = shard_pairs[proj_name][2]
                p0 = getattr(mlp0, proj_name)
                p1 = getattr(mlp1, proj_name)
                a0 = p0.lora_A[_ADAPTER].weight
                b0 = p0.lora_B[_ADAPTER].weight
                if kind == "col":
                    n0 = int(p0.base_layer.host_weight.weight.shape[0])
                    _copy_param(a0, orig_a)
                    _copy_param(b0, orig_b[:n0])
                    _copy_param(p1.lora_B[_ADAPTER].weight, orig_b[n0:])
                    mirror = _demote_to_mirror(p1.lora_A[_ADAPTER], a0, dev1)
                else:
                    k0 = int(p0.base_layer.host_weight.weight.shape[1])
                    _copy_param(b0, orig_b)
                    _copy_param(a0, orig_a[:, :k0])
                    _copy_param(p1.lora_A[_ADAPTER].weight, orig_a[:, k0:])
                    mirror = _demote_to_mirror(p1.lora_B[_ADAPTER], b0, dev1)
                mirror_pairs.append((b0 if kind == "row" else a0, mirror))
                param_map[f"{layer_tag}.mlp.{proj_name}"] = {"kind": kind}

        # branch1 SDPA saved tensors must offload like branch0's (installed by name on the
        # original self_attn parents); attn1 is text attention by construction, install direct.
        install_attention_saved_tensor_offload(attn1)

        # ---- branch1 norms + assembly ----
        norm1_in = _shared_frozen_norm(layer.input_layernorm)
        norm1_post = _shared_frozen_norm(layer.post_attention_layernorm)

        stp_layer = StpDecoderLayer(
            self_attn=attn,
            mlp=mlp0,
            input_layernorm=layer.input_layernorm,
            post_attention_layernorm=layer.post_attention_layernorm,
            self_attn_stp1=attn1,
            mlp_stp1=mlp1,
            input_layernorm_stp1=norm1_in,
            post_attention_layernorm_stp1=norm1_post,
        )
        layers[layer_idx] = stp_layer
        counts["layers"] += 1
        rt.stp_dev1_modules_wrapped += 1

    setattr(model, "_asym_stp_mirror_pairs", mirror_pairs)
    setattr(model, "_asym_stp_param_map", param_map)
    setattr(model, "_asym_stp_full_tp", True)
    # NOTE: finalize_stp_placement runs LATER (end of apply_lf_asym_lora) — the residency
    # validator expects the not-yet-moved CPU state it sees under |1.
    return counts


def finalize_stp_placement(model: nn.Module, dev0: torch.device) -> None:
    """The HF Trainer's blanket _move_model_to_device(cuda:0) would flatten branch1 onto
    dev0, so: (a) advertise model parallelism (Trainer then skips the move entirely) and
    (b) finish the placement ourselves — every still-CPU Parameter/buffer (embeddings,
    rotary inv_freq, branch0 norms, final norm) moves to dev0. Deliberately-CPU state is
    untouched: HostWeight arenas and bias_cpu are plain attrs, not Parameters/buffers;
    branch1 params already live on dev1."""
    moved = 0
    for module_name, module in model.named_modules():
        for param_name, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                raise RuntimeError(f"meta parameter survived the sTP build: {module_name}.{param_name}")
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                raise RuntimeError(f"meta buffer survived the sTP build: {module_name}.{buffer_name}")
    model.is_parallelizable = True
    model.model_parallel = True
    setattr(model, "_asym_stp_placement_moved", moved)


def stp_load_adapter_init(model: nn.Module, path: str) -> int:
    """Transplant a |1 run's adapter INIT dump into the sTP branch pieces (parity gates:
    RNG draw order differs between |1 and the branch construction, so step-2 grads only
    compare when the values match). Full A/B tensors map onto branch slices per the
    param_map; mirrors are refreshed from their owners afterwards."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reference = payload.get("params", payload)
    param_map = getattr(model, "_asym_stp_param_map", {})
    named = dict(model.named_parameters())
    loaded = 0
    if not param_map:
        # Phase-A / plain |1 structure: names match one-to-one
        with torch.no_grad():
            for name, value in reference.items():
                target = named.get(name)
                if target is not None and tuple(target.shape) == tuple(value.shape):
                    target.copy_(value.to(target.device, target.dtype))
                    loaded += 1
        return loaded
    for logical, meta in param_map.items():
        kind = meta["kind"]
        branch1_logical = logical.replace(".self_attn.", ".self_attn_stp1.").replace(".mlp.", ".mlp_stp1.")
        for piece in ("lora_A", "lora_B"):
            ref_key = f"{logical}.{piece}.{_ADAPTER}.weight"
            if ref_key not in reference:
                continue
            full = reference[ref_key]
            own_key = f"{logical}.{piece}.{_ADAPTER}.weight"
            other_key = f"{branch1_logical}.{piece}.{_ADAPTER}.weight"
            own = named.get(own_key)
            other = named.get(other_key)
            sliced = (kind == "col" and piece == "lora_B") or (kind == "row" and piece == "lora_A")
            with torch.no_grad():
                if sliced and own is not None and other is not None:
                    if kind == "col":
                        n0 = int(own.shape[0])
                        own.copy_(full[:n0].to(own.device, own.dtype))
                        other.copy_(full[n0:].to(other.device, other.dtype))
                    else:
                        k0 = int(own.shape[1])
                        own.copy_(full[:, :k0].to(own.device, own.dtype))
                        other.copy_(full[:, k0:].to(other.device, other.dtype))
                    loaded += 2
                elif own is not None:
                    own.copy_(full.to(own.device, own.dtype))
                    loaded += 1
    with torch.no_grad():
        for owner, mirror in getattr(model, "_asym_stp_mirror_pairs", []):
            mirror.data.copy_(owner.data.to(mirror.device))
    return loaded


def stp_post_backward_merge(model: nn.Module) -> int:
    """Fold every mirror's partial grad into its owner param (dA = dA0 + to0(dA1);
    row-dB likewise). Call AFTER backward completes (training_step return), BEFORE
    clip/optimizer. Mirrors' grads are cleared so nothing double-counts."""
    pairs = getattr(model, "_asym_stp_mirror_pairs", None)
    if not pairs:
        return 0
    rt = get_runtime()
    merged = 0
    for owner, mirror in pairs:
        grad = mirror.grad
        if grad is None:
            continue
        moved = rt.to0(grad) if grad.device != owner.device else grad
        if owner.grad is None:
            owner.grad = moved.clone() if moved is grad else moved
        else:
            owner.grad.add_(moved)
        mirror.grad = None
        merged += 1
    return merged


def stp_register_optimizer_resync(model: nn.Module, optimizer) -> None:
    """After each optimizer step, refresh every mirror from its (just-updated) owner."""
    pairs = getattr(model, "_asym_stp_mirror_pairs", None)
    if not pairs:
        return
    # accelerate's AcceleratedOptimizer proxies register_step_post_hook but lacks the
    # underlying hook dict; unwrap to the real torch optimizer (AsymCPUAdamW).
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    if getattr(optimizer, "_asym_stp_resync_hooked", False):
        return

    def _resync(_opt, _args, _kwargs) -> None:
        with torch.no_grad():
            for owner, mirror in pairs:
                mirror.data.copy_(owner.data, non_blocking=True)

    optimizer.register_step_post_hook(_resync)
    optimizer._asym_stp_resync_hooked = True
