#!/usr/bin/env bash
set -Eeuo pipefail

# Create a normal python -m venv under AsymGEMM, with bin/activate.
# This is separate from the existing LlamaFactory/.venv conda-prefix env.

# Repo root = AsymGEMM-SFT (../.. from the AsymGEMM dir you run in). Override with SFT_ROOT=...
SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ASYMGEMM_DIR=${ASYMGEMM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-${SFT_ROOT}/third_party/LlamaFactory}
KT_DIR=${KT_DIR:-${SFT_ROOT}/third_party/ktransformers}
DEEPSPEED_DIR=${DEEPSPEED_DIR:-${SFT_ROOT}/third_party/deepspeed}
LIGER_DIR=${LIGER_DIR:-${SFT_ROOT}/third_party/Liger-Kernel}
ENV_DIR=${ENV_DIR:-${ASYMGEMM_DIR}/.venv}
PYTHON_BIN=${PYTHON_BIN:-python3}

RECREATE_ENV=${RECREATE_ENV:-0}
INSTALL_LF=${INSTALL_LF:-1}
INSTALL_KT=${INSTALL_KT:-1}
INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED:-1}
INSTALL_ASYMGEMM=${INSTALL_ASYMGEMM:-0}
INSTALL_KT_KERNEL=${INSTALL_KT_KERNEL:-0}
INSTALL_LIGER=${INSTALL_LIGER:-1}
INSTALL_FLA=${INSTALL_FLA:-1}
INSTALL_CAUSAL_CONV1D=${INSTALL_CAUSAL_CONV1D:-1}

# Torch stack, pinned to the known-good venv (torch 2.12.0 built against CUDA 13.0).
# Override any var to retarget CUDA/versions. A non-empty TORCH_INSTALL_CMD wins
# over the pinned default below (e.g. an air-gapped wheel dir or another channel).
TORCH_VERSION=${TORCH_VERSION:-2.12.0+cu130}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.27.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}
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
else
  # Default: pinned torch/vision/audio from the CUDA 13.0 wheel index, installed
  # before LF so its resolver treats torch as already satisfied and won't swap it.
  # If torch 2.12.0 has moved off the stable channel, point TORCH_INDEX_URL at
  # https://download.pytorch.org/whl/test/cu130 (or the nightly channel) instead.
  python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
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

if [[ "${INSTALL_LIGER}" == "1" ]]; then
  # Liger-Kernel fused Triton kernels. LF's enable_liger_kernel/apply_liger_kernel
  # path and asym_gemm.integrations.liger_loss both import this, so it is required
  # for the normal training path. Editable from the local checkout; --no-deps keeps
  # it from dragging torch/triton/transformers off the versions pinned above.
  python -m pip install --no-deps -e "${LIGER_DIR}"
fi

if [[ "${INSTALL_FLA}" == "1" ]]; then
  # flash-linear-attention (+ fla-core) for linear-/gated-attention model paths.
  # Pure Triton/Python; transformers is already satisfied by LF above, so only
  # fla-core gets added.
  python -m pip install "flash-linear-attention==0.5.0" "fla-core==0.5.0"
fi

if [[ "${INSTALL_CAUSAL_CONV1D}" == "1" ]]; then
  # causal-conv1d CUDA kernels for Mamba/hybrid blocks. Build against this venv's
  # torch instead of an isolated build env. Set INSTALL_CAUSAL_CONV1D=0 if the host
  # has no matching CUDA build toolchain.
  python -m pip install --no-build-isolation "causal_conv1d==1.6.2.post1"
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
try:
    import liger_kernel
    print("liger_kernel", getattr(liger_kernel, "__version__", "ok"))
except Exception as exc:
    print("liger_kernel import failed:", repr(exc))
try:
    import fla
    print("flash-linear-attention", getattr(fla, "__version__", "unknown"))
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
