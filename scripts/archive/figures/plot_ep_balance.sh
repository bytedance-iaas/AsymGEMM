#!/usr/bin/env bash
# Driver for the per-model EP-effectiveness figures.
#
# One two-panel figure per MoE model (Expert GEMM | MoE Block), four z-skew
# clusters x four bars (EP / sDP / sEP plan / sEP queue), mean over 3 seeded
# shuffles with min..max caps. Rotated % labels: gray = EP GPU-imbalance,
# red = reduction vs EP, dark blue = reduction vs sDP (keyed in-panel).
# Emits four figures sharing the same style:
#   ep_balance_q330b.{pdf,png}    Qwen3-30B-A3B
#   ep_balance_q3235b.{pdf,png}   Qwen3-235B-A22B
#   ep_balance_q35122b.{pdf,png}  Qwen3.5-122B-A10B
#   ep_balance_l4scout.{pdf,png}  Llama-4-Scout
#
# Numbers are the banked ep_balance_bench sweeps (profiling_results/profiling_both_skew/, see
# MODELS in plot_ep_balance.py). Fonts/sizes/colors/layout via constants.py.
#
# Usage: ./plot_ep_balance.sh [OUTPUT_DIR] [--model q3-30b-a3b|...|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/out}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_ep_balance.py" --output-dir "${OUTPUT_DIR}" "${@:2}"
