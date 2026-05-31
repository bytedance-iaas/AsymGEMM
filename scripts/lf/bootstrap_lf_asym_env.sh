#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# User Parameters
# =============================================================================
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}
LF_DIR=${LF_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory}
LF_REPO_URL=${LF_REPO_URL:-https://github.com/hiyouga/LLaMA-Factory.git}
LF_BRANCH=${LF_BRANCH:-main_kevin}
ASYM_BRANCH=${ASYM_BRANCH:-main_kevin}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
RECREATE_ENV=${RECREATE_ENV:-0}
TORCH_INSTALL_CMD=${TORCH_INSTALL_CMD:-}
CONDA_EXE=${CONDA_EXE:-conda}

# =============================================================================
# Derived Parameters
# =============================================================================
ASYM_DIR=${ASYM_DIR:-${ROOT}}
ENV_DIR=${ENV_DIR:-${LF_DIR}/.venv}

# =============================================================================
# Main Logic
# =============================================================================
checkout_branch_or_keep() {
  local repo_dir=$1
  local branch=$2
  if git -C "${repo_dir}" rev-parse --verify "${branch}" >/dev/null 2>&1; then
    git -C "${repo_dir}" checkout "${branch}"
    return
  fi
  if git -C "${repo_dir}" ls-remote --exit-code --heads origin "${branch}" >/dev/null 2>&1; then
    git -C "${repo_dir}" fetch origin "${branch}"
    git -C "${repo_dir}" checkout -b "${branch}" "origin/${branch}"
    return
  fi
  echo "Warning: branch ${branch} not found for ${repo_dir}; keeping current branch $(git -C "${repo_dir}" branch --show-current)." >&2
}

if [[ ! -d "${LF_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${LF_DIR}")"
  if git ls-remote --exit-code --heads "${LF_REPO_URL}" "${LF_BRANCH}" >/dev/null 2>&1; then
    git clone -b "${LF_BRANCH}" "${LF_REPO_URL}" "${LF_DIR}"
  else
    git clone "${LF_REPO_URL}" "${LF_DIR}"
  fi
fi

checkout_branch_or_keep "${LF_DIR}" "${LF_BRANCH}"

if [[ -d "${ASYM_DIR}/.git" ]]; then
  checkout_branch_or_keep "${ASYM_DIR}" "${ASYM_BRANCH}"
fi

if [[ "${RECREATE_ENV}" == "1" && -d "${ENV_DIR}" ]]; then
  "${CONDA_EXE}" env remove -y -p "${ENV_DIR}"
fi

if [[ ! -d "${ENV_DIR}" ]]; then
  "${CONDA_EXE}" create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}"
fi

"${CONDA_EXE}" run -p "${ENV_DIR}" python -m pip install -U pip wheel packaging ninja
if [[ -n "${TORCH_INSTALL_CMD}" ]]; then
  "${CONDA_EXE}" run -p "${ENV_DIR}" bash -lc "${TORCH_INSTALL_CMD}"
fi
"${CONDA_EXE}" run -p "${ENV_DIR}" python -m pip install -e "${LF_DIR}"
"${CONDA_EXE}" run -p "${ENV_DIR}" python -m pip install --no-build-isolation -e "${ASYM_DIR}"

"${CONDA_EXE}" run -p "${ENV_DIR}" python - <<'PY'
import torch
import asym_gemm

print("torch", torch.__version__)
print("asym_gemm", asym_gemm.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
