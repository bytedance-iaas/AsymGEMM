#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User Parameters
# =============================================================================
ROOT=${ROOT:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM}

DEVICE="${DEVICE:-cuda:0}"
TOKENS="${TOKENS:-2048}"
IN_FEATURES="${IN_FEATURES:-2048}"
OUT_FEATURES="${OUT_FEATURES:-768}"
RANK="${RANK:-16}"
SCALE="${SCALE:-16}"
DTYPE="${DTYPE:-bf16}"
PRECISION="${PRECISION:-bf16}"
ASYM_BF16_OUTPUT_DTYPE="${ASYM_BF16_OUTPUT_DTYPE:-fp32}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
ATOL="${ATOL:-5e-2}"
RTOL="${RTOL:-5e-2}"
L2_RTOL="${L2_RTOL:-5e-3}"
SEED="${SEED:-123}"
ZERO_B="${ZERO_B:-false}"
REQUIRE_ASYM_CALLS="${REQUIRE_ASYM_CALLS:-true}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

# =============================================================================
# Derived Parameters
# =============================================================================
CHECK_SCRIPT="${ROOT}/scripts/lora/check_lora_ops.py"

# =============================================================================
# Main Logic
# =============================================================================
usage() {
  cat <<USAGE
Usage:
  scripts/lora/check_lora_ops.sh [extra check_lora_ops.py args]

Environment/user parameters:
  ROOT=${ROOT}
  DEVICE=${DEVICE}
  TOKENS=${TOKENS}
  IN_FEATURES=${IN_FEATURES}
  OUT_FEATURES=${OUT_FEATURES}
  RANK=${RANK}
  SCALE=${SCALE}
  DTYPE=${DTYPE}
  PRECISION=${PRECISION}
  ASYM_BF16_OUTPUT_DTYPE=${ASYM_BF16_OUTPUT_DTYPE}
  ACCUM_STEPS=${ACCUM_STEPS}
  ATOL=${ATOL}
  RTOL=${RTOL}
  L2_RTOL=${L2_RTOL}
  SEED=${SEED}
  ZERO_B=${ZERO_B}
  REQUIRE_ASYM_CALLS=${REQUIRE_ASYM_CALLS}

Examples:
  scripts/lora/check_lora_ops.sh
  TOKENS=256 IN_FEATURES=1024 OUT_FEATURES=1024 ACCUM_STEPS=2 scripts/lora/check_lora_ops.sh
USAGE
}

bool_arg() {
  case "${1,,}" in
    1|true|yes|y|on) printf '%s\n' "--$2" ;;
    0|false|no|n|off) printf '%s\n' "--no-$2" ;;
    *) echo "error: expected boolean for $2, got '${1}'" >&2; exit 2 ;;
  esac
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

cmd=(
  "${PYTHON_BIN}" "${CHECK_SCRIPT}"
  --device "${DEVICE}"
  --tokens "${TOKENS}"
  --in-features "${IN_FEATURES}"
  --out-features "${OUT_FEATURES}"
  --rank "${RANK}"
  --scale "${SCALE}"
  --dtype "${DTYPE}"
  --precision "${PRECISION}"
  --asym-bf16-output-dtype "${ASYM_BF16_OUTPUT_DTYPE}"
  --accum-steps "${ACCUM_STEPS}"
  --atol "${ATOL}"
  --rtol "${RTOL}"
  --l2-rtol "${L2_RTOL}"
  --seed "${SEED}"
)
cmd+=("$(bool_arg "${ZERO_B}" "zero-b")")
cmd+=("$(bool_arg "${REQUIRE_ASYM_CALLS}" "require-asym-calls")")
cmd+=("$@")

printf '+'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
