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

# Query-chunked attention BACKWARD for qwen3.5's full-attention layers.
#
# At long seq the global HBM peak is one GC layer's attention re-backward working
# set (~70.9 GiB of the 78.9 GiB peak at s80000). This wrapper
# leaves the forward untouched (one fused FA4 call) but computes the backward in
# query blocks: for q rows [s0:s1) causal attention only reads keys [0:s1), so a
# chunk re-forward over (q_chunk, k[:s1], v[:s1]) + torch.autograd.grad yields the
# exact dq chunk and partial dk/dv, accumulated in fp32. Peak per moment becomes
# k/v + one chunk's tensors + dk/dv accumulators instead of the whole layer set.
# Cost: one extra chunked forward inside backward (~+1/3 attention FLOPs).
#
# Model-level and backend-agnostic (fairness, same policy as qwen35_delta_chunk).
# Env: QWEN35_ATTN_BWD_CHUNK_Q = query tokens per chunk (0/unset = off).
# Bottom-right causal alignment for seqlen_q < seqlen_k is the flash-attn
# convention and is what makes the prefix-key chunking exact; the parity probe
# guards it.

import os

import torch

from ...extras import logging


logger = logging.get_logger(__name__)

_WRAPPED_ATTR = "_lf_attn_chunk_wrapped"


def _chunk_q() -> int:
    raw = os.environ.get("QWEN35_ATTN_BWD_CHUNK_Q", "")
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


class _ChunkedBwdAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, orig_fn, module, mask, scaling, kwargs):
        with torch.no_grad():
            out, _ = orig_fn(module, q, k, v, mask, scaling=scaling, **kwargs)
        ctx.save_for_backward(q, k, v)
        ctx.orig_fn = orig_fn
        ctx.module = module
        ctx.scaling = scaling
        ctx.kwargs = kwargs
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        chunk = _chunk_q()
        seq_q = int(q.shape[2])  # layouts: [B, H, S, D]
        dq = torch.empty_like(q)
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        for s0 in range(0, seq_q, chunk):
            s1 = min(s0 + chunk, seq_q)
            qc = q[:, :, s0:s1].detach().contiguous().requires_grad_(True)
            kc = k[:, :, :s1].detach().contiguous().requires_grad_(True)
            vc = v[:, :, :s1].detach().contiguous().requires_grad_(True)
            with torch.enable_grad():
                out_c, _ = ctx.orig_fn(ctx.module, qc, kc, vc, None, scaling=ctx.scaling, **ctx.kwargs)
            gq, gk, gv = torch.autograd.grad(out_c, (qc, kc, vc), dout[:, s0:s1].contiguous())
            dq[:, :, s0:s1] = gq
            dk[:, :, :s1] += gk.to(torch.float32)
            dv[:, :, :s1] += gv.to(torch.float32)
            del qc, kc, vc, out_c, gq, gk, gv
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def _wrap(orig_fn):
    def wrapped(module, query, key, value, attention_mask, scaling=None, **kwargs):
        chunk = _chunk_q()
        if attention_mask is not None and chunk > 0:
            # An all-valid mask is equivalent to no mask for the dense causal
            # path; padded batches keep their mask and fall through to the
            # original varlen handling.
            try:
                dense = bool(attention_mask.all()) if attention_mask.dtype == torch.bool else bool((attention_mask != 0).all())
            except Exception:
                dense = False
            if dense:
                attention_mask = None
        if (
            chunk <= 0
            or not torch.is_grad_enabled()
            or not module.training
            or int(query.shape[2]) <= chunk
            or attention_mask is not None
            or kwargs.get("dropout", 0.0)
        ):
            return orig_fn(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
        out = _ChunkedBwdAttn.apply(query, key, value, orig_fn, module, attention_mask, scaling, kwargs)
        return out, None

    wrapped._lf_attn_chunk_original = orig_fn  # type: ignore[attr-defined]
    return wrapped


def apply_qwen35_attn_chunk(model: "torch.nn.Module") -> bool:
    """Wrap the model's attention implementation with the chunked-backward path.

    Inert unless QWEN35_ATTN_BWD_CHUNK_Q > 0 at forward time. Registry-level wrap
    of the resolved attention function, restricted to qwen3.5 model types.
    """
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if not model_type.startswith("qwen3_5"):
        return False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return False
    impl = str(getattr(model.config, "_attn_implementation", "") or "")
    fn = ALL_ATTENTION_FUNCTIONS.get(impl)
    if fn is None or getattr(fn, "_lf_attn_chunk_original", None) is not None:
        return fn is not None
    ALL_ATTENTION_FUNCTIONS[impl] = _wrap(fn)
    logger.info_rank0(
        f"qwen3.5 chunked attention backward wrapped over '{impl}' (QWEN35_ATTN_BWD_CHUNK_Q={_chunk_q() or 'off'})."
    )
    return True
