# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sequence-chunked Qwen3.5 gated-deltanet forward (training/prefill path).
#
# Why: at long sequences the stock Qwen3_5(Moe)GatedDeltaNet.forward materializes
# the whole layer's fla chunk-state stack at once (~30 GiB at s45000 b8), and
# fla's chunk_gated_delta_rule kernel is broken above ~70k tokens/row on fla
# 0.5.0/0.5.1 (CUDA illegal memory access standalone; silent loss-0/NaN inside a
# full model). Processing the sequence in chunks with a differentiable
# recurrent-state carry is mathematically exact for the linear recurrence (bf16
# accumulation differences only, measured ~0.5% rel) and keeps every fla call
# below the fault length.
#
# This is a MODEL-level, backend-agnostic patch (scoreboard fairness: every
# training backend gets identical math). Gated by QWEN35_DELTA_CHUNK_SIZE
# (tokens per chunk; unset/0 = stock behavior, re-read per forward call).
# Decode/cached paths (cache_params is not None) always use the original code.
# Each chunk optionally runs under non-reentrant gradient checkpointing so only
# one chunk's graph is live during backward; that part auto-disables under
# DeepSpeed ZeRO-3, whose partitioned (0-numel) parameters trip torch's
# recompute metadata validation (QWEN35_DELTA_CHUNK_CHECKPOINT=0 forces it off).
# Carries between chunks: the fla recurrent state and the (kernel_size-1)
# pre-conv tail columns (exact causal-conv continuation).
#
# Validated by a kernel-level parity probe and loss-band identity against
# unchunked runs at 45k tokens.

import os
from functools import partial

import torch
import torch.nn.functional as F

from ...extras import logging


logger = logging.get_logger(__name__)

_ORIG_ATTR = "_lf_delta_chunk_original_forward"


def _chunk_size() -> int:
    raw = os.environ.get("QWEN35_DELTA_CHUNK_SIZE", "")
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


def _chunk_local_saves_enabled() -> bool:
    # QWEN35_DELTA_CHUNK_LOCAL_SAVES=1: shadow any outer save_on_cpu hooks inside
    # each chunk's checkpoint so chunk-internal saves stay on GPU (one chunk's
    # worth at a time) instead of CPU round-trips whose unpacked copies were
    # measured stacking to ~70 GiB at s80000.
    return os.environ.get("QWEN35_DELTA_CHUNK_LOCAL_SAVES", "0").strip().lower() in {"1", "true", "yes", "on"}


def _chunk_checkpoint_enabled() -> bool:
    raw = os.environ.get("QWEN35_DELTA_CHUNK_CHECKPOINT", "1").lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            return False
    except Exception:
        pass
    return True


def _gdn_chunk_step(module, apply_mask_fn, h_c, m_c, recurrent_state, conv_tail):
    """One sequence chunk of the stock GatedDeltaNet.forward math."""
    h_c = apply_mask_fn(h_c, m_c)
    batch_size, seq_len, _ = h_c.shape
    kernel = module.conv_kernel_size

    mixed_qkv = module.in_proj_qkv(h_c).transpose(1, 2)  # [B, conv_dim, L]
    z = module.in_proj_z(h_c).reshape(batch_size, seq_len, -1, module.head_v_dim)
    b = module.in_proj_b(h_c)
    a = module.in_proj_a(h_c)

    new_tail = mixed_qkv[:, :, -(kernel - 1):]
    if conv_tail is None:
        conv_in, drop = mixed_qkv, 0
    else:
        conv_in, drop = torch.cat([conv_tail, mixed_qkv], dim=-1), kernel - 1

    if module.causal_conv1d_fn is not None:
        conv_out = module.causal_conv1d_fn(
            x=conv_in,
            weight=module.conv1d.weight.squeeze(1),
            bias=module.conv1d.bias,
            activation=module.activation,
            seq_idx=None,
        )
        if drop:
            conv_out = conv_out[:, :, drop:]
    else:
        conv_out = F.silu(module.conv1d(conv_in)[:, :, drop : drop + seq_len])

    mixed = conv_out.transpose(1, 2)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)

    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    if module.num_v_heads // module.num_k_heads > 1:
        query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
        key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)

    core, next_state = module.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    core = core.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core = module.norm(core, z)
    core = core.reshape(batch_size, seq_len, -1)
    return module.out_proj(core), next_state, new_tail


def _make_chunked_forward(apply_mask_fn):
    def _chunked_forward(self, hidden_states, cache_params=None, attention_mask=None):
        from torch.utils.checkpoint import checkpoint

        chunk = _chunk_size()
        _, seq_len, _ = hidden_states.shape
        original = getattr(type(self), _ORIG_ATTR)
        if cache_params is not None or chunk <= 0 or seq_len <= chunk or chunk < self.conv_kernel_size:
            return original(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)

        step = partial(_gdn_chunk_step, self, apply_mask_fn)
        use_ckpt = torch.is_grad_enabled() and self.training and _chunk_checkpoint_enabled()
        outputs = []
        recurrent_state = None
        conv_tail = None
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            h_c = hidden_states[:, start:end]
            if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape[-1] == seq_len:
                m_c = attention_mask[:, start:end]
            else:
                m_c = attention_mask
            if use_ckpt:
                if _chunk_local_saves_enabled():
                    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                        out_c, recurrent_state, conv_tail = checkpoint(
                            step, h_c, m_c, recurrent_state, conv_tail, use_reentrant=False
                        )
                else:
                    out_c, recurrent_state, conv_tail = checkpoint(
                        step, h_c, m_c, recurrent_state, conv_tail, use_reentrant=False
                    )
            else:
                out_c, recurrent_state, conv_tail = step(h_c, m_c, recurrent_state, conv_tail)
            outputs.append(out_c)
        return torch.cat(outputs, dim=1)

    return _chunked_forward


def apply_qwen35_delta_chunk(model: "torch.nn.Module") -> int:
    """Install the chunked forward on every Qwen3.5 GatedDeltaNet class in `model`.

    Class-level, idempotent. Inert unless QWEN35_DELTA_CHUNK_SIZE > 0 at forward
    time (the env is re-read per call, so runs flip it without a reload).
    Returns the number of linear_attn module instances covered.
    """
    covered = 0
    patched: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith("linear_attn"):
            continue
        if not hasattr(module, "chunk_gated_delta_rule") or not hasattr(module, "conv_kernel_size"):
            continue
        cls = type(module)
        if not hasattr(cls, _ORIG_ATTR):
            modeling = __import__(cls.__module__, fromlist=["apply_mask_to_padding_states"])
            apply_mask_fn = getattr(modeling, "apply_mask_to_padding_states", None)
            if apply_mask_fn is None:
                continue
            setattr(cls, _ORIG_ATTR, cls.forward)
            cls.forward = _make_chunked_forward(apply_mask_fn)
            patched.append(cls.__name__)
        covered += 1
    if covered:
        logger.info_rank0(
            f"Qwen3.5 delta-net chunked forward installed on {covered} linear_attn modules "
            f"(classes={patched or 'already patched'}, QWEN35_DELTA_CHUNK_SIZE={_chunk_size() or 'off'})."
        )
    return covered
