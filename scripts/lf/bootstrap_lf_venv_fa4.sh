#!/usr/bin/env bash
set -Eeuo pipefail

# Build the qwen3.5/FlashAttention-4 LlamaFactory runtime in a sibling env.
# This intentionally does not touch ${ASYMGEMM_DIR}/.venv.

SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ASYMGEMM_DIR=${ASYMGEMM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory-fa4}
KT_DIR=${KT_DIR:-${SFT_ROOT}/third_party/ktransformers}
DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
LIGER_DIR=${LIGER_DIR:-${SFT_ROOT}/third_party/Liger-Kernel}
ENV_DIR=${ENV_DIR:-${ASYMGEMM_DIR}/.venv-fa4}

if [[ -z "${PYTHON_BIN+x}" ]]; then
  if [[ -x "${LF_DIR}/.conda-lf-fa4/bin/python" ]]; then
    PYTHON_BIN="${LF_DIR}/.conda-lf-fa4/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  else
    PYTHON_BIN=python3
  fi
fi

RECREATE_ENV=${RECREATE_ENV:-0}
INSTALL_LF=${INSTALL_LF:-1}
INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED:-1}
INSTALL_KT=${INSTALL_KT:-1}
INSTALL_LIGER=${INSTALL_LIGER:-1}
INSTALL_FLA=${INSTALL_FLA:-1}
# Optional qwen3.5 linear-attention conv acceleration. The model has a torch
# fallback for this piece, while flash-linear-attention still provides the
# fused gated-delta kernels. Keep this opt-in because the upstream build script
# currently compiles many CUDA architectures even when TORCH_CUDA_ARCH_LIST is
# restricted to SM100.
INSTALL_CAUSAL_CONV1D=${INSTALL_CAUSAL_CONV1D:-0}
INSTALL_ASYMGEMM=${INSTALL_ASYMGEMM:-1}
INSTALL_KT_KERNEL=${INSTALL_KT_KERNEL:-0}

# Pinned to the locally validated LlamaFactory-fa4 environment.
TORCH_VERSION=${TORCH_VERSION:-2.12.0+cu130}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.27.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.6.0}
FLASH_ATTN4_VERSION=${FLASH_ATTN4_VERSION:-4.0.0b16}
NVIDIA_CUTLASS_DSL_VERSION=${NVIDIA_CUTLASS_DSL_VERSION:-4.5.2}
FLASH_LINEAR_ATTENTION_VERSION=${FLASH_LINEAR_ATTENTION_VERSION:-0.5.0}
FLA_CORE_VERSION=${FLA_CORE_VERSION:-0.5.0}
CAUSAL_CONV1D_VERSION=${CAUSAL_CONV1D_VERSION:-1.6.2.post1}
CAUSAL_CONV1D_TORCH_CUDA_ARCH_LIST=${CAUSAL_CONV1D_TORCH_CUDA_ARCH_LIST:-10.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}
TORCH_INSTALL_CMD=${TORCH_INSTALL_CMD:-}

if [[ ! -d "${LF_DIR}" ]]; then
  echo "Missing FA4 LlamaFactory checkout: ${LF_DIR}" >&2
  exit 2
fi

if [[ "${RECREATE_ENV}" == "1" && -d "${ENV_DIR}" ]]; then
  rm -rf "${ENV_DIR}"
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv --prompt asymgemm-lf-fa4 "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

python -m pip install -U pip "setuptools<82" wheel packaging ninja

if [[ -n "${TORCH_INSTALL_CMD}" ]]; then
  bash -lc "${TORCH_INSTALL_CMD}"
else
  python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
fi

python -m pip install \
  "transformers==${TRANSFORMERS_VERSION}" \
  "nvidia-cutlass-dsl==${NVIDIA_CUTLASS_DSL_VERSION}" \
  "nvidia-cutlass-dsl-libs-cu13==${NVIDIA_CUTLASS_DSL_VERSION}" \
  "flash-attn-4==${FLASH_ATTN4_VERSION}"

if [[ "${INSTALL_LF}" == "1" ]]; then
  python -m pip install -e "${LF_DIR}"
fi

if [[ "${INSTALL_DEEPSPEED}" == "1" ]]; then
  DS_BUILD_OPS=${DS_BUILD_OPS:-0} python -m pip install --no-build-isolation -e "${DEEPSPEED_DIR}"
fi

if [[ "${INSTALL_KT}" == "1" ]]; then
  python -m pip install --no-deps -e "${KT_DIR}"
fi

if [[ "${INSTALL_ASYMGEMM}" == "1" ]]; then
  python -m pip install --no-build-isolation -e "${ASYMGEMM_DIR}"
fi

if [[ "${INSTALL_KT_KERNEL}" == "1" ]]; then
  (cd "${KT_DIR}/kt-kernel" && bash ./install.sh build)
fi

if [[ "${INSTALL_LIGER}" == "1" ]]; then
  python -m pip install --no-deps -e "${LIGER_DIR}"
fi

if [[ "${INSTALL_FLA}" == "1" ]]; then
  python -m pip install \
    "flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}" \
    "fla-core==${FLA_CORE_VERSION}"
fi

if [[ "${INSTALL_CAUSAL_CONV1D}" == "1" ]]; then
  TORCH_CUDA_ARCH_LIST="${CAUSAL_CONV1D_TORCH_CUDA_ARCH_LIST}" \
    python -m pip install --no-build-isolation "causal_conv1d==${CAUSAL_CONV1D_VERSION}"
fi

if [[ -f "${LF_DIR}/fa4_probes/patch_transformers_fa4.py" ]]; then
  python "${LF_DIR}/fa4_probes/patch_transformers_fa4.py"
fi

python - <<'PY'
import importlib.metadata as md
import inspect
import sys

print("python", sys.executable)

import torch
import transformers
from transformers.utils import is_flash_attn_4_available
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("flash_attn_4_available", is_flash_attn_4_available())
print("flash_linear_attention_available", is_flash_linear_attention_available())
print("causal_conv1d_available", is_causal_conv1d_available())
print("flash_attn_4", md.version("flash-attn-4"))
print("nvidia_cutlass_dsl", md.version("nvidia-cutlass-dsl"))

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
print("flash_attn_func", flash_attn_func)
print("flash_attn_varlen_func", flash_attn_varlen_func)

import transformers.integrations.flash_attention as hf_flash_attention
src = inspect.getsource(hf_flash_attention.flash_attention_forward)
if "s_aux=s_aux.to(query.dtype) if s_aux is not None else None" not in src:
    raise RuntimeError("Transformers FA4 wrapper is missing the s_aux=None fix")

try:
    import llamafactory
    print("llamafactory", getattr(llamafactory, "__version__", "unknown"), llamafactory.__file__)
except Exception as exc:
    print("llamafactory import failed:", repr(exc))

try:
    import fla
    print("flash-linear-attention", getattr(fla, "__version__", md.version("flash-linear-attention")))
except Exception as exc:
    print("flash-linear-attention import failed:", repr(exc))

try:
    import causal_conv1d
    print("causal_conv1d", getattr(causal_conv1d, "__version__", "unknown"))
except Exception as exc:
    print("causal_conv1d import failed:", repr(exc))
PY

echo
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
