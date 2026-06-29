#!/usr/bin/env bash
# Driver for the LoRA SFT step-time breakdown figure.
#
# Generates a grouped stacked-bar PDF of forward/backward/optimizer time for
# Qwen3-32B and Qwen3-MoE-30B under ZeRO-3 Offload vs SuperOffload, motivating
# that the optimizer step (SuperOffload's only target) is not the bottleneck.
#
# NOTE: uses illustrative PLACEHOLDER timings -- see the BARS block in
# plot_lora_timing.py to swap in measured numbers.
#
# Usage: ./plot_lora_timing.sh [OUTPUT_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/out}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_lora_timing.py" --output-dir "${OUTPUT_DIR}"
