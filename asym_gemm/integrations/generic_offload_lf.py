"""Backend-agnostic activation-offload BASELINES for LlamaFactory (run on non-asym backends).

The AsymGEMM LF integration (``apply_lf_asym_lora``) only runs when ``use_asym_gemm=true`` (asym backends);
``zero3_offload`` / ``superoffload_mem`` never invoke it. These baselines therefore live in a separate,
backend-agnostic hook (``maybe_apply_generic_offload``), called at the end of LlamaFactory's ``init_adapter``.

Two baselines, selected purely from the harness env (no extra flag):

* ``off-layer``  -> ``apply_off_layer``: wrap each decoder layer forward in ``torch.autograd.graph.save_on_cpu``
  (offload every saved activation to CPU, no recompute). The generic "full offload" floor.
* ``none|T|T|F|T|T`` (any of expert_act/attn_act/layer_gc set) -> ``apply_generic_same_policy``: the same policy
  AsymGEMM uses, but offload via the *generic* ``save_on_cpu`` kernel (glue-GC ``generic_offload=True``) plus
  sdpa-recompute. Isolates AsymGEMM's custom offload kernel from the policy.

No-op unless a baseline is requested; heavy deps are imported lazily.
"""

from __future__ import annotations

import os

import torch
from torch import nn


def _env_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _log_rank0(message: str) -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(f"[generic_offload_lf] {message}", flush=True)


def _install_save_on_cpu(module: nn.Module) -> None:
    """Run ``module.forward`` under ``save_on_cpu`` so all its saved activations offload to CPU."""
    if getattr(module, "_asym_generic_save_on_cpu_installed", False):
        return
    original_forward = module.forward

    def forward(*args, _orig=original_forward, **kwargs):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return _orig(*args, **kwargs)

    module.forward = forward  # type: ignore[method-assign]
    setattr(module, "_asym_generic_save_on_cpu_installed", True)


def apply_off_layer(model: nn.Module) -> int:
    """Baseline #1 (floor): ``save_on_cpu`` around each decoder layer; offload everything, no recompute."""
    from asym_gemm.integrations.lf import _is_qwen3_decoder_layer_module_name

    wrapped = 0
    for name, module in list(model.named_modules()):
        if name and _is_qwen3_decoder_layer_module_name(name, module):
            _install_save_on_cpu(module)
            wrapped += 1
    if wrapped == 0:
        raise RuntimeError("off-layer baseline: no supported decoder layers matched")
    return wrapped


def apply_generic_same_policy(model: nn.Module) -> tuple[int, int, int]:
    """Baseline #2: offload ONLY attn + expert activations (module-scoped ``save_on_cpu``), recompute norms
    (glue-GC, ``offload_mode="none"`` -> no whole-layer offload) + sdpa. Matches AsymGEMM ``none|T|T|F|T|T``'s
    offload SCOPE (exp+attn only, nothing outside) with a generic kernel -- NOT off-layer (whole-layer)."""
    from asym_gemm.integrations.lf import (
        _is_qwen3_decoder_layer_module_name,
        _is_text_attention_module_name,
    )
    from asym_gemm.training.decoder_layer_glue_gc import install_decoder_layer_glue_gc
    from asym_gemm.training.sdpa_recompute import install_sdpa_recompute

    n_attn = n_exp = n_layer = 0
    # Attention: scoped save_on_cpu (offload ONLY attention activations).
    for name, module in list(model.named_modules()):
        if name and _is_text_attention_module_name(name, module):
            _install_save_on_cpu(module)
            n_attn += 1
    # Per decoder layer: offload its FFN child (mlp / feed_forward = the MoE/expert block) via save_on_cpu, and
    # recompute norms via glue-GC (offload_mode="none" -> NO whole-layer offload, so nothing outside attn+FFN
    # is offloaded). Matching by the FFN child is robust to the stock-HF / LoRA-wrapped expert format (the
    # fused-format is_qwen3_experts matcher does not match it on the superoffload path).
    for name, module in list(model.named_modules()):
        if not (name and _is_qwen3_decoder_layer_module_name(name, module)):
            continue
        ffn = getattr(module, "mlp", None)
        if not isinstance(ffn, nn.Module):
            ffn = getattr(module, "feed_forward", None)
        if isinstance(ffn, nn.Module):
            _install_save_on_cpu(ffn)
            n_exp += 1
        install_decoder_layer_glue_gc(module, offload_mode="none")
        n_layer += 1
    if n_attn + n_exp == 0:
        raise RuntimeError("generic same-policy baseline: no attention/FFN modules matched")
    install_sdpa_recompute(model)  # self-gates on ASYMM_ATTN_SDPA_RECOMPUTE
    return n_attn, n_exp, n_layer


def maybe_apply_generic_offload(model: nn.Module) -> None:
    """End-of-``init_adapter`` hook (non-asym backends). Auto-selects a baseline from the harness env; no-op otherwise.

    Reached only on non-asym backends (the asym branch returns early in ``init_adapter``), so any offload the policy
    requests here is applied generically. No dedicated flag -- detected from the policy + ``ASYMM_*`` offload envs.
    """
    policy = os.environ.get("ASYM_GEMM_LF_CONFIG_EXPERT_POLICY", "none").strip()
    # Some harness paths carry 'none' in the CONFIG var while the canonical ASYM_EXPERT_RECOMPUTE_POLICY
    # still holds 'off-layer'. Prefer the explicit off-layer signal so the floor baseline isn't lost.
    if policy != "off-layer" and os.environ.get("ASYM_EXPERT_RECOMPUTE_POLICY", "").strip() == "off-layer":
        policy = "off-layer"
    exp = _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD"))
    attn = _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD"))
    layer_gc = _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC"))
    _log_rank0(
        f"hook reached: EXPERT_POLICY={policy!r} exp={exp} attn={attn} layer_gc={layer_gc} "
        f"(ASYM_EXPERT_RECOMPUTE_POLICY={os.environ.get('ASYM_EXPERT_RECOMPUTE_POLICY')!r})"
    )

    if policy == "off-layer":
        count = apply_off_layer(model)
        _log_rank0(f"off-layer baseline: wrapped {count} decoder layers in save_on_cpu (no recompute)")
    elif exp or attn or layer_gc:
        n_attn, n_exp, n_layer = apply_generic_same_policy(model)
        _log_rank0(
            f"generic same-policy baseline: save_on_cpu on {n_attn} attn + {n_exp} expert modules "
            f"(offload scope = exp+attn ONLY), glue-GC norm-recompute on {n_layer} layers + sdpa-recompute "
            f"(exp={exp} attn={attn} layer_gc={layer_gc})"
        )
    else:
        return


__all__ = ["maybe_apply_generic_offload", "apply_off_layer", "apply_generic_same_policy"]
