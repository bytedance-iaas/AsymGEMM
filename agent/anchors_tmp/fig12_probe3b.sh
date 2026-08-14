#!/bin/bash
# fig12 probe 3b — frontier pairs for the figure: same-day A arms + B(staged)
# at the pressure-zone lengths. Serial, GPU0. W1+M1 (fit/latency protocol:
# step-2 is the measured step; variance at these cells ~1%).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
export MAX_STEPS=1 WARMUP_STEPS=1
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|true|false|false|false"
SYS="asym_cpuadamwds|T3"

echo "PROBE3B begin $(date +%H:%M)" >> "$S"
# same-day A at 1.6M (pairs with pfs1600)
ARM_ENV=""                          run_cell pfa1600 q3-30b-a3b "$SYS" 1600000 "1" "$POL" 1
# 1.4M pair (pressure-zone probe: B ~86-90% HBM expected)
ARM_ENV="ASYMM_LORA_KERNELS=staged" run_cell pfs1400 q3-30b-a3b "$SYS" 1400000 "1" "$POL" 1
ARM_ENV=""                          run_cell pfa1400 q3-30b-a3b "$SYS" 1400000 "1" "$POL" 1
echo "PROBE3B-DONE $(date +%H:%M)" >> "$S"
