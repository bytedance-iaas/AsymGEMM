#!/usr/bin/env bash
set -Eeuo pipefail

# Create .venv-nemo-fa4 — the FlashAttention-4 sibling of .venv-nemo, mirroring
# the .venv / .venv-fa4 pairing of the LF stack (bootstrap_lf_venv_fa4.sh).
#
# Layers ON TOP of a full bootstrap_nemo_venv.sh install (run into this env's
# directory) the qwen3.5/FA4 attention stack at the SAME pins as the LF fa4
# env: flash-attn-4 + cutlass-dsl, and the 0.5.x flash-linear-attention
# kernels (the base nemo env pins fla 0.4.2 = Megatron-Bridge lock vintage;
# the fa4 env upgrades to the 0.5.0 gated-delta kernels the LF fa4 env runs).
# Megatron/TE picks its own attention backend (cuDNN fused on sm100); FA4 here
# is for import-parity with the LF fa4 stack and any fla path that wants it.
#
# Run INSIDE the asym_sft container. Idempotent; RECREATE_ENV=1 wipes.

SFT_ROOT=${SFT_ROOT:-$(cd ../.. && pwd)}
ASYMGEMM_DIR=${ASYMGEMM_DIR:-${SFT_ROOT}/third_party/AsymGEMM}
ENV_DIR=${ENV_DIR:-${ASYMGEMM_DIR}/.venv-nemo-fa4}

# Pinned to the locally validated FA4 (.venv-fa4) environment.
FLASH_ATTN4_VERSION=${FLASH_ATTN4_VERSION:-4.0.0b16}
NVIDIA_CUTLASS_DSL_VERSION=${NVIDIA_CUTLASS_DSL_VERSION:-4.5.2}
FLA_VERSION=${FLA_VERSION:-0.5.0}
NVIDIA_CUBLAS_VERSION=${NVIDIA_CUBLAS_VERSION:-13.6.1.10}

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full nemo baseline env into .venv-nemo-fa4 (same pins except the fla series).
SFT_ROOT="${SFT_ROOT}" ASYMGEMM_DIR="${ASYMGEMM_DIR}" ENV_DIR="${ENV_DIR}" \
FLA_VERSION="${FLA_VERSION}" NVIDIA_CUBLAS_VERSION="${NVIDIA_CUBLAS_VERSION}" \
RECREATE_ENV="${RECREATE_ENV:-0}" \
  bash "${_here}/bootstrap_nemo_venv.sh"

source "${ENV_DIR}/bin/activate"

# FA4 layer (same package set as bootstrap_lf_venv_fa4.sh).
python -m pip install \
  "nvidia-cutlass-dsl==${NVIDIA_CUTLASS_DSL_VERSION}" \
  "nvidia-cutlass-dsl-libs-cu13==${NVIDIA_CUTLASS_DSL_VERSION}" \
  "flash-attn-4==${FLASH_ATTN4_VERSION}"

# The FA4 layer's resolution can touch torch and downgrade cublas again
# (see bootstrap_nemo_venv.sh); re-assert the TE-compatible pin LAST.
python -m pip install "nvidia-cublas==${NVIDIA_CUBLAS_VERSION}"

python - <<'PY'
import sys

failures = []
print("python", sys.executable)

import torch  # noqa: E402  (preloads venv cublas for TE)

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())

for label, mod in [
    ("transformer_engine.pytorch", "transformer_engine.pytorch"),
    ("megatron.bridge", "megatron.bridge"),
    ("flash_attn (FA4)", "flash_attn"),
    ("flash-linear-attention", "fla"),
]:
    try:
        import importlib

        m = importlib.import_module(mod)
        print(label, getattr(m, "__version__", "ok"))
    except Exception as exc:
        print(f"{label} import failed [REQUIRED]:", repr(exc))
        failures.append(label)

try:
    import fla

    assert fla.__version__.startswith("0.5"), f"fla {fla.__version__} != 0.5.x"
    print("fla series ok:", fla.__version__)
except Exception as exc:
    print("fla version check failed [REQUIRED]:", repr(exc))
    failures.append("fla-0.5.x")

if failures:
    print("ERROR: fa4 layer verification failed:", ", ".join(failures), file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
