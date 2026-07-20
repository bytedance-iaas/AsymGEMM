#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default paper figure source. Specify only the plot artifact; plot_c2c.py
# infers metrics/interconnect/data/timeseries.csv from this PNG path.
export C2C_INPUT_PLOT="${C2C_INPUT_PLOT:-${REPO_ROOT}/profiling_results/profiling_both/asym_long_sft_smoke__lora__lf__bf16/llama-3_3-70b-instruct__gpus1__b8_s32000_ga1_w1_s1_r64_a16_drop000/superoffload_mem__nsys__unsloth-off__polnone__routerhf__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__moefg0__dscatter0__route000_lora0_accfp32__ligerloss1/b8_s32000_ga1/metrics/interconnect/timeline_max.png}"

if [[ $# -eq 0 || "${1}" == --* ]]; then
  set -- "${C2C_INPUT_PLOT}" "$@"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_c2c.py" "$@"
