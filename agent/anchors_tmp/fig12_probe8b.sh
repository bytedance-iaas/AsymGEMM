#!/bin/bash
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-500}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
export MAX_STEPS=1 WARMUP_STEPS=1
POLN="none|false|false|false|false|false"
OFF="ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true"
echo "PROBE8B begin $(date +%H:%M)" >> "$S"
ARM_ENV="$OFF" run_cell hyuo192 hunyuan-a13b "asym_cpuadamwds|unsloth-off" 192000 "1" "$POLN" 1
echo "PROBE8B-DONE $(date +%H:%M)" >> "$S"
