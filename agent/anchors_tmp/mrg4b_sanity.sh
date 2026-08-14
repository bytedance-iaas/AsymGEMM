#!/bin/bash
# mrg4b sanity (2026-08-14 4-way merge): venv imports + merged-module checks +
# fg numeric probes, extended from mrg4_sanity.sh with the mrg4b surface
# (ep_sep count_skip graft, frozen_linear §3.1a-c union + ALLOW_DENSE,
# SFT fig12 ASYMM_LORA_KERNELS arms, rebuilt _C after the ep_steal assert
# relaxations). Runs INSIDE the enroot container (asym_sft_45, SFT-38 tree).
set -uo pipefail
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
R=agent/anchors_tmp/mrg4b_sanity_report.txt
: > "$R"
for v in .venv .venv-fa4; do
  echo "== venv $v ==" >> "$R"
  "$v/bin/python" - >> "$R" 2>&1 <<'PY'
import os
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "dev_count", torch.cuda.device_count())
import asym_gemm
from asym_gemm import _C
print("bf16 asym binding:", hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"))
print("ep_steal binding:", hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal"))
# mrg4b: sepplanlink2 union surface
from asym_gemm.training import ep_sep
st_cls = ep_sep._SepState
print("ep_sep count_skip graft:", hasattr(st_cls, "count_skip"))
print("ep_sep pre_gate:", hasattr(st_cls, "pre_gate"))
print("ep_sep transports:", sorted(ep_sep._MAX_MPE_DEFAULTS))
# NOTE: asym_gemm.training.__init__ re-exports a FUNCTION named frozen_linear
# that shadows the submodule on `from ... import` — go through sys.modules.
import sys
import asym_gemm.training.frozen_linear
fl = sys.modules["asym_gemm.training.frozen_linear"]
print("frozen_linear _try_ep_sep_grouped:", hasattr(fl, "_try_ep_sep_grouped"))
print("frozen_linear ALLOW_DENSE flag:", hasattr(fl, "_EP_SEP_ALLOW_DENSE"))
# mrg4b: SFT fig12 kernel-ablation arms (env-gated, default off)
from asym_gemm.training import cpu_left
print("cpu_left lora_kernels_mode default:", repr(cpu_left._lora_kernels_mode()))
import asym_gemm.training.exp_act_offload_lora
# mrg4 (08-09) surface, re-verified
import asym_gemm.training.jamba_moe as jm
print("jamba_moe import OK:", jm.AsymJambaMoeBlock._is_asym_jamba_moe_block)
import asym_gemm.training.attention_activation_offload
import asym_gemm.training.activation_offload
from asym_gemm.training import exact_pinned
import asym_gemm.training.host_weight
import asym_gemm.integrations.lf as lf
import asym_gemm.integrations.liger_loss
print("classify mamba:", lf.classify_lf_component("model.layers.3.mamba"))
print("classify rotary:", lf.classify_lf_component("model.layers.3.self_attn.rotary_emb"))
from liger_kernel.transformers.monkey_patch import (
    apply_liger_kernel_to_jamba,
    apply_liger_kernel_to_glm4_moe,
    apply_liger_kernel_to_glm4_moe_lite,
)
print("liger appliers OK")
import mamba_ssm
print("mamba_ssm", mamba_ssm.__version__)
import llamafactory.model.loader
import llamafactory.model.patcher
from llamafactory.model.model_utils.liger_kernel import _resolve_liger_apply_fn
print("LF jamba resolver:", _resolve_liger_apply_fn("jamba") is not None)
print("VENV-SANITY-PASS", flush=True)
PY
done
echo "== numeric probe hunyuan (fg101 vs fp32) ==" >> "$R"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --hunyuan --tokens 2048 >> "$R" 2>&1
echo "== numeric probe mixtral ==" >> "$R"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --mixtral --tokens 2048 >> "$R" 2>&1
echo "== numeric probe qwen3 ==" >> "$R"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --qwen3 --tokens 2048 >> "$R" 2>&1
echo "SANITY-DONE" >> "$R"
