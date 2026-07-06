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
  echo "       Install Python ${REQUIRED_PYTHON_VERSION} (or set PYTHON_BIN to a ${REQUIRED_PYTHON_VERSION} interpreter) and re-run." >&2
  exit 1
fi

RECREATE_ENV=${RECREATE_ENV:-0}
INSTALL_LF=${INSTALL_LF:-1}
INSTALL_KT=${INSTALL_KT:-1}
INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED:-1}
# asym_gemm is imported unconditionally by run_lf_profiled_train.py (the profiling
# entrypoint that runs against this .venv), and asym_gemm/__init__.py resolves its
# version via importlib.metadata, which requires the package to be pip-installed
# (importable-from-source is not enough). Install it by default; set =0 to skip.
INSTALL_ASYMGEMM=${INSTALL_ASYMGEMM:-1}
INSTALL_KT_KERNEL=${INSTALL_KT_KERNEL:-0}
INSTALL_LIGER=${INSTALL_LIGER:-1}
INSTALL_FLA=${INSTALL_FLA:-1}
INSTALL_CAUSAL_CONV1D=${INSTALL_CAUSAL_CONV1D:-1}
# Provision the libaio sidecar (${ASYMGEMM_DIR}/.aioenv) that DeepSpeed NVMe offload
# (*opnvme/*panvme backends) needs. Idempotent; set SETUP_AIO=0 to skip.
SETUP_AIO=${SETUP_AIO:-1}

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

if [[ "${SETUP_AIO}" == "1" ]]; then
  # Provision ${ASYMGEMM_DIR}/.aioenv (libaio.h + libaio.so) so DeepSpeed can JIT-build the
  # async_io op that the NVMe-offload backends need. run_lf_lora_sft.sh auto-detects this
  # sidecar and exports CPATH/LIBRARY_PATH/LD_LIBRARY_PATH at run time. Idempotent; verify
  # only when DeepSpeed is installed (the check imports AsyncIOBuilder). Fatal on failure —
  # this env is for NVMe-offload runs, so a broken sidecar means a broken env; pass
  # SETUP_AIO=0 to deliberately skip on a host that cannot build it (air-gapped, no curl).
  AIO_SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_aioenv.sh"
  aio_args=(); [[ "${INSTALL_DEEPSPEED}" == "1" ]] || aio_args=(--no-verify)
  if ! ROOT="${ASYMGEMM_DIR}" ENV_PYTHON="${ENV_DIR}/bin/python" \
        bash "${AIO_SETUP}" "${aio_args[@]}"; then
    echo "ERROR: setup_aioenv.sh failed — libaio sidecar not provisioned." >&2
    echo "       DeepSpeed NVMe *opnvme/*panvme backends require it. Fix the failure above," >&2
    echo "       or re-run the bootstrap with SETUP_AIO=0 to intentionally skip it." >&2
    exit 1
  fi
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

INSTALL_LF="${INSTALL_LF}" INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED}" \
INSTALL_LIGER="${INSTALL_LIGER}" INSTALL_FLA="${INSTALL_FLA}" \
INSTALL_CAUSAL_CONV1D="${INSTALL_CAUSAL_CONV1D}" \
INSTALL_ASYMGEMM="${INSTALL_ASYMGEMM}" \
python - <<'PY'
import os
import sys

# A package that was requested (INSTALL_*=1) but fails to import means a broken env:
# record it and exit non-zero at the end. Packages not requested are only reported.
failures = []
print("python", sys.executable)

# torch is the base of the whole stack — always required.
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed [REQUIRED]:", repr(exc))
    failures.append("torch")

for label, mod, flag in [
    ("llamafactory", "llamafactory", "INSTALL_LF"),
    ("deepspeed", "deepspeed", "INSTALL_DEEPSPEED"),
    ("liger_kernel", "liger_kernel", "INSTALL_LIGER"),
    ("flash-linear-attention", "fla", "INSTALL_FLA"),
    ("causal_conv1d", "causal_conv1d", "INSTALL_CAUSAL_CONV1D"),
    # asym_gemm: importing it triggers importlib.metadata.version('asym_gemm'),
    # so this fails at bootstrap if the package was not actually pip-installed.
    ("asym_gemm", "asym_gemm", "INSTALL_ASYMGEMM"),
]:
    required = os.environ.get(flag, "0") == "1"
    try:
        m = __import__(mod)
        print(label, getattr(m, "__version__", "ok"))
    except Exception as exc:
        print("%s import failed [%s]:" % (label, "REQUIRED" if required else "optional/not-requested"), repr(exc))
        if required:
            failures.append(label)

if failures:
    print("ERROR: requested packages failed to import:", ", ".join(failures), file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
