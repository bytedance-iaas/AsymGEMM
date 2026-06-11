#!/usr/bin/env bash
set -Eeuo pipefail

# Create a normal python -m venv under AsymGEMM, with bin/activate.
# This is separate from the existing LlamaFactory/.venv conda-prefix env.

SFT_ROOT=${SFT_ROOT:-/home/kevinni/AsymGEMM-SFT}
ASYMGEMM_DIR=${ASYMGEMM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
KT_DIR=${KT_DIR:-${SFT_ROOT}/third_party/ktransformers}
DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
ENV_DIR=${ENV_DIR:-${ASYMGEMM_DIR}/.venv}
PYTHON_BIN=${PYTHON_BIN:-python3}

RECREATE_ENV=${RECREATE_ENV:-0}
INSTALL_LF=${INSTALL_LF:-1}
INSTALL_KT=${INSTALL_KT:-1}
INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED:-1}
INSTALL_ASYMGEMM=${INSTALL_ASYMGEMM:-0}
INSTALL_KT_KERNEL=${INSTALL_KT_KERNEL:-0}
TORCH_INSTALL_CMD=${TORCH_INSTALL_CMD:-}

if [[ "${RECREATE_ENV}" == "1" && -d "${ENV_DIR}" ]]; then
  rm -rf "${ENV_DIR}"
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv --prompt asymgemm-lf "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

python -m pip install -U pip "setuptools<82" wheel packaging ninja

if [[ -n "${TORCH_INSTALL_CMD}" ]]; then
  bash -lc "${TORCH_INSTALL_CMD}"
fi

if [[ "${INSTALL_LF}" == "1" ]]; then
  # Install LlamaFactory normally so pip resolves the app/runtime dependencies.
  python -m pip install -e "${LF_DIR}"
fi

if [[ "${INSTALL_DEEPSPEED}" == "1" ]]; then
  # Build DeepSpeed against this venv's torch/CUDA stack instead of an isolated build env.
  DS_BUILD_OPS=${DS_BUILD_OPS:-0} python -m pip install --no-build-isolation -e "${DEEPSPEED_DIR}"
fi

if [[ "${INSTALL_KT}" == "1" ]]; then
  # Skip ktransformers dependencies so kt-kernel comes from local source, not PyPI.
  python -m pip install --no-deps -e "${KT_DIR}"
fi

if [[ "${INSTALL_ASYMGEMM}" == "1" ]]; then
  # Build AsymGEMM against this venv's torch/CUDA stack instead of an isolated build env.
  python -m pip install --no-build-isolation -e "${ASYMGEMM_DIR}"
fi

if [[ "${INSTALL_KT_KERNEL}" == "1" ]]; then
  # Build kt-kernel from its local source tree when explicitly requested.
  (cd "${KT_DIR}/kt-kernel" && bash ./install.sh build)
fi

python - <<'PY'
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed:", repr(exc))
try:
    import llamafactory
    print("llamafactory", getattr(llamafactory, "__version__", "unknown"))
except Exception as exc:
    print("llamafactory import failed:", repr(exc))
try:
    import deepspeed
    print("deepspeed", getattr(deepspeed, "__version__", "unknown"))
except Exception as exc:
    print("deepspeed import failed:", repr(exc))
PY

echo
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
