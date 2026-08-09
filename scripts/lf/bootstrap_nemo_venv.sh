#!/usr/bin/env bash
set -Eeuo pipefail

# Create .venv-nemo — the NeMo / Megatron-Bridge BASELINE env (Megatron-LM with
# LoRA support) used by scripts/lf/profile_lora_nemo.sh. Sibling of
# bootstrap_lf_venv.sh: same interpreter pin, same torch/CUDA pin, but a fully
# separate venv so the nemo baseline never contaminates the LF/AsymGEMM stack
# (nemo needs transformers 5.8.x vs LF's 5.6.0).
#
# Run INSIDE the asym_sft_NN enroot container (python 3.12.3, CUDA 13.0 + nvcc,
# cuDNN 9 w/ headers, GB200 sm100, aarch64). Idempotent; RECREATE_ENV=1 wipes.
#
# What goes in (versions = Megatron-Bridge uv.lock at checkout dabf51d9d unless
# noted):
#   torch 2.12.0+cu130 (house pin, same as .venv/.venv-fa4; lock has 2.12.1)
#   transformer-engine[pytorch,core_cu13] 2.16.0 — the lock pins the same 2.16.0
#     vintage from git; the PyPI release ships a prebuilt manylinux aarch64
#     core_cu13 wheel, so only the (small) torch-binding sdist compiles here.
#   megatron-core = the vendored 3rdparty/Megatron-LM submodule, EDITABLE
#     (uv.lock treats it the same way; --no-deps + explicit core deps).
#   megatron-bridge itself, EDITABLE, --no-deps + the curated runtime closure
#     below (import megatron.bridge eagerly imports every model bridge, so the
#     closure must cover vl/audio/diffusion bridge imports too).
#   flash-linear-attention 0.4.2 (qwen3.5 / qwen3-next delta-net path) +
#     causal-conv1d (their short-conv kernels; built against this torch).

# Repo root = the AsymGEMM-SFT* workspace (../.. from the AsymGEMM dir you run in). Override with SFT_ROOT=...
SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ASYMGEMM_DIR=${ASYMGEMM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
MBRIDGE_DIR=${MBRIDGE_DIR:-${SFT_ROOT}/third_party/Megatron-Bridge}
MBRIDGE_GIT_URL=${MBRIDGE_GIT_URL:-https://github.com/NVIDIA-NeMo/Megatron-Bridge.git}
ENV_DIR=${ENV_DIR:-${ASYMGEMM_DIR}/.venv-nemo}

# Pinned interpreter: Python 3.12.3 is the only allowed/validated version for this env.
REQUIRED_PYTHON_VERSION=${REQUIRED_PYTHON_VERSION:-3.12.3}
if [[ -z "${PYTHON_BIN+x}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
  else
    PYTHON_BIN=python3
  fi
fi
_py_ver="$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
if [[ "${_py_ver}" != "${REQUIRED_PYTHON_VERSION}" ]]; then
  echo "ERROR: this environment requires Python ${REQUIRED_PYTHON_VERSION}, but PYTHON_BIN=${PYTHON_BIN} reports '${_py_ver:-not found}'." >&2
  echo "       Run inside the asym_sft container (or set PYTHON_BIN to a ${REQUIRED_PYTHON_VERSION} interpreter) and re-run." >&2
  exit 1
fi

RECREATE_ENV=${RECREATE_ENV:-0}
INSTALL_TE=${INSTALL_TE:-1}
INSTALL_CAUSAL_CONV1D=${INSTALL_CAUSAL_CONV1D:-1}

# Torch stack, pinned to the known-good house venvs (torch 2.12.0 / CUDA 13.0).
TORCH_VERSION=${TORCH_VERSION:-2.12.0+cu130}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.27.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}
TORCH_INSTALL_CMD=${TORCH_INSTALL_CMD:-}

# Baseline stack pins (Megatron-Bridge uv.lock vintage).
TE_VERSION=${TE_VERSION:-2.16.0}
# TE 2.16's prebuilt aarch64 core lib references cublasLt grouped-GEMM symbols
# added after the CUDA 13.0/13.1 cublas that torch 2.12.0+cu130 pins; the 13.6
# pip cublas provides them and stays within the forward-compatible .so.13 ABI.
NVIDIA_CUBLAS_VERSION=${NVIDIA_CUBLAS_VERSION:-13.6.1.10}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.8.1}
PEFT_VERSION=${PEFT_VERSION:-0.19.1}
DATASETS_VERSION=${DATASETS_VERSION:-5.0.0}
ACCELERATE_VERSION=${ACCELERATE_VERSION:-1.14.0}
FLA_VERSION=${FLA_VERSION:-0.4.2}
NVRX_VERSION=${NVRX_VERSION:-0.6.0}
MODELOPT_VERSION=${MODELOPT_VERSION:-0.44.0}
CAUSAL_CONV1D_VERSION=${CAUSAL_CONV1D_VERSION:-1.6.2.post1}
NVIDIA_PYPI_URL=${NVIDIA_PYPI_URL:-https://pypi.nvidia.com}

# Megatron-Bridge checkout: clone if absent so the script is reproducible from
# a bare workspace; never mutate an existing checkout (submodule init aside).
if [[ ! -d "${MBRIDGE_DIR}" ]]; then
  git clone "${MBRIDGE_GIT_URL}" "${MBRIDGE_DIR}"
fi
if [[ ! -f "${MBRIDGE_DIR}/3rdparty/Megatron-LM/pyproject.toml" ]]; then
  (cd "${MBRIDGE_DIR}" && git submodule update --init 3rdparty/Megatron-LM)
fi

if [[ "${RECREATE_ENV}" == "1" && -d "${ENV_DIR}" ]]; then
  rm -rf "${ENV_DIR}"
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv --prompt asymgemm-nemo "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

python -m pip install -U pip "setuptools<80" wheel packaging ninja pybind11

if [[ -n "${TORCH_INSTALL_CMD}" ]]; then
  bash -lc "${TORCH_INSTALL_CMD}"
else
  python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
fi

if [[ "${INSTALL_TE}" == "1" ]]; then
  # transformer_engine core lands as a prebuilt manylinux aarch64 cu13 wheel;
  # only transformer-engine-torch (the pybind glue) builds here, against this
  # venv's torch + the container's cuDNN headers — hence no build isolation.
  python -m pip install --no-build-isolation \
    "transformer_engine[pytorch,core_cu13]==${TE_VERSION}"
fi

# megatron-core = the vendored submodule, editable. --no-deps so pip cannot
# resolve the heavyweight [dev,mlm] extras; its real core deps (torch/numpy/
# packaging) are already present or installed below.
python -m pip install --no-build-isolation --no-deps -e "${MBRIDGE_DIR}/3rdparty/Megatron-LM"

# megatron-bridge itself, editable, --no-deps: the closure below is the curated
# runtime dependency set (bridge pyproject minus inference/CI-only packages:
# flashinfer, mlflow, comet-ml, nemo-run, modelopt).
python -m pip install --no-deps -e "${MBRIDGE_DIR}"

python -m pip install \
  "transformers==${TRANSFORMERS_VERSION}" \
  "peft==${PEFT_VERSION}" \
  "datasets==${DATASETS_VERSION}" \
  "accelerate==${ACCELERATE_VERSION}" \
  "flash-linear-attention==${FLA_VERSION}" \
  "fla-core==${FLA_VERSION}" \
  "omegaconf==2.3.1" \
  "hydra-core==1.3.2" \
  "einops==0.8.2" \
  "mistral-common" \
  "qwen-vl-utils" \
  "diffusers==0.38.0" \
  "timm" \
  "open-clip-torch" \
  "rich" \
  "tensorboard" \
  "wandb" \
  "sentencepiece" \
  "tiktoken" \
  "regex" \
  "six" \
  "pyyaml" \
  "tqdm" \
  "typing-extensions" \
  "numpy" \
  "psutil"

# nvidia-resiliency-ext: mcore's fault-tolerance hooks import it in the train
# loop; NVIDIA publishes the aarch64 wheel on its own index.
# nvidia-modelopt: imported unconditionally by megatron.bridge's distillation
# provider, so it is part of the import closure (0.44.0 has no 'torch' extra).
python -m pip install \
  "nvidia-resiliency-ext==${NVRX_VERSION}" \
  "nvidia-modelopt==${MODELOPT_VERSION}" \
  --extra-index-url "${NVIDIA_PYPI_URL}"

if [[ "${INSTALL_CAUSAL_CONV1D}" == "1" ]]; then
  # Short-conv CUDA kernels for the qwen3.5/qwen3-next linear-attention blocks.
  python -m pip install --no-build-isolation "causal_conv1d==${CAUSAL_CONV1D_VERSION}"
fi

# megatron-energon: mcore's dataloader package, imported by the bridge training
# path. It drags fsspec past datasets' ceiling, so re-pin fsspec right after.
python -m pip install "megatron-energon~=7.0"
python -m pip install "fsspec[http]<=2026.4.0"

if [[ "${INSTALL_TE}" == "1" ]]; then
  # MUST BE THE LAST INSTALL. torch 2.12.0+cu130 metadata hard-pins
  # nvidia-cublas==13.1.1.3.*, so every later dependency resolution that touches
  # torch silently downgrades cublas; TE's prebuilt lib needs the grouped-GEMM
  # cublasLt symbols that first appear in the newer 13.x series (see
  # NVIDIA_CUBLAS_VERSION above). Installing it after everything else wins the
  # fight; the resulting pip resolver warning is expected and benign (the .so.13
  # minor series is forward-compatible). The verify block imports TE (after
  # torch) which asserts the symbol actually resolves.
  python -m pip install "nvidia-cublas==${NVIDIA_CUBLAS_VERSION}"
fi

INSTALL_TE="${INSTALL_TE}" INSTALL_CAUSAL_CONV1D="${INSTALL_CAUSAL_CONV1D}" \
python - <<'PY'
import os
import sys

failures = []
print("python", sys.executable)

try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed [REQUIRED]:", repr(exc))
    failures.append("torch")

for label, mod, flag in [
    ("transformer_engine", "transformer_engine", "INSTALL_TE"),
    ("transformer_engine.pytorch", "transformer_engine.pytorch", "INSTALL_TE"),
    ("megatron.core", "megatron.core", "ALWAYS"),
    ("megatron.bridge", "megatron.bridge", "ALWAYS"),
    ("transformers", "transformers", "ALWAYS"),
    ("peft", "peft", "ALWAYS"),
    ("flash-linear-attention", "fla", "ALWAYS"),
    ("causal_conv1d", "causal_conv1d", "INSTALL_CAUSAL_CONV1D"),
    ("nvidia_resiliency_ext", "nvidia_resiliency_ext", "ALWAYS"),
]:
    required = flag == "ALWAYS" or os.environ.get(flag, "0") == "1"
    try:
        import importlib
        m = importlib.import_module(mod)
        print(label, getattr(m, "__version__", "ok"))
    except Exception as exc:
        print("%s import failed [%s]:" % (label, "REQUIRED" if required else "optional/not-requested"), repr(exc))
        if required:
            failures.append(label)

# The two campaign bridges must be registered for AutoBridge dispatch.
if "megatron.bridge" not in failures:
    try:
        from megatron.bridge import AutoBridge  # noqa: F401
        import megatron.bridge.models.qwen.qwen3_moe_bridge  # noqa: F401
        import megatron.bridge.models.qwen.qwen35_bridge  # noqa: F401
        print("qwen3-moe + qwen3.5 bridges ok")
    except Exception as exc:
        print("qwen bridge import failed [REQUIRED]:", repr(exc))
        failures.append("qwen-bridges")

if failures:
    print("ERROR: requested packages failed to import:", ", ".join(failures), file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
