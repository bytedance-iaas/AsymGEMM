#!/usr/bin/env bash
# Driver for the main AsymLoRA memory-saving figure(s).
#
# Grouped vertical bars, one group per model (labeled with model name + total
# tokens per step), one bar per system (SuperOffload / SuperOffload + Act. Offload /
# AsymLoRA (Ours)), styled after the reference throughput figure. Emits two
# companion figures sharing the same style:
#   memory_saving_hbm.{pdf,png}   Peak HBM (GiB), lower = better (the saving)
#   memory_saving_dram.{pdf,png}  Host DRAM (GiB), the offloading trade-off cost
#
# Numbers are measured (see MODELS in plot_main.py / paper.md Results table).
# Fonts/sizes/colors/layout are shared via constants.py.
#
# Usage: ./plot_main.sh [OUTPUT_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/out}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_main.py" --output-dir "${OUTPUT_DIR}"
