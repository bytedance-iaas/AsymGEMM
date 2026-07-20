#!/usr/bin/env bash
# Driver for the LoRA SFT memory-breakdown figure.
#
# Grouped stacked bars of peak memory (Model Weights / LoRA Weights / LoRA Grads
# / Activations) per model, one bar per sequence length (64k, 128k).
#
# NOTE: uses illustrative PLACEHOLDER memory numbers -- see the BARS block in
# plot_memory.py to swap in measured GiB.
#
# Usage: ./plot_memory.sh [OUTPUT_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/out}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_memory.py" --output-dir "${OUTPUT_DIR}"
