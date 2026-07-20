#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
OVERLEAF_FIG_DIR="${REPO_ROOT}/agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA/figures"

mkdir -p "${OVERLEAF_FIG_DIR}"

cp "${OUT_DIR}/lora_timing_breakdown.pdf" "${OVERLEAF_FIG_DIR}/optimizer_timing.pdf"
cp "${OUT_DIR}/memory_breakdown.pdf" "${OVERLEAF_FIG_DIR}/memory_decomposition.pdf"
cp "${OUT_DIR}/c2c_timeline.pdf" "${OVERLEAF_FIG_DIR}/c2c_rx_utilization.pdf"
cp "${OUT_DIR}/utilization_timeline.pdf" "${OVERLEAF_FIG_DIR}/cpu_utilization.pdf"

echo "copied 4 figure PDFs to ${OVERLEAF_FIG_DIR}"
