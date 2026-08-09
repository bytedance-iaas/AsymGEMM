#!/bin/bash
# 4way-merge sanity (2026-08-09): venv imports + merged-module checks + fg numeric probes.
# Runs INSIDE the enroot container (asym_sft_45, SFT-38 tree mounted).
set -uo pipefail
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
R=agent/anchors_tmp/mrg4_sanity_report.txt
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
import asym_gemm.training.jamba_moe as jm
print("jamba_moe import OK:", jm.AsymJambaMoeBlock._is_asym_jamba_moe_block)
import asym_gemm.training.attention_activation_offload  # glm lora-a reroute (46)
import asym_gemm.training.activation_offload            # exact-saved pool (SFT)
from asym_gemm.training import exact_pinned
import asym_gemm.training.host_weight                   # clone slab (SFT)
import asym_gemm.integrations.lf as lf
import asym_gemm.integrations.liger_loss
os.environ.pop("ASYM_EXACT_PINNED_SAVED", None)
a = exact_pinned.exact_saved_enabled()
os.environ["ASYM_EXACT_PINNED_SAVED"] = "1"
b = exact_pinned.exact_saved_enabled()
print("exact_saved flag unset/set:", a, b)
print("classify mamba:", lf.classify_lf_component("model.layers.3.mamba"))
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
