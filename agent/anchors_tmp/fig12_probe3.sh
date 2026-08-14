#!/bin/bash
# fig12 probe 3 — capacity frontier: does the kernels-off (staged swap-backs)
# arm OOM where AsymLoRA runs? 30B T3 b1 at 1.6M -> 1.4M -> 1.2M.
# Fit/no-fit protocol: W1+M1. Early-exit: first length that FITS ends the walk
# (wall bracketed).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
export MAX_STEPS=1 WARMUP_STEPS=1
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|true|false|false|false"
SYS="asym_cpuadamwds|T3"

echo "PROBE3 begin $(date +%H:%M)" >> "$S"
for skk in 1600 1400 1200; do
  v=$(ARM_ENV="ASYMM_LORA_KERNELS=staged" run_cell "pfs${skk}" q3-30b-a3b "$SYS" "${skk}000" "1" "$POL" 1)
  echo "PROBE3 pfs${skk} -> $v" >> "$S"
  if [ "$v" = "TRAINED" ]; then
    echo "PROBE3 staged-arm FITS at ${skk}k — wall bracketed above" >> "$S"
    break
  fi
done
echo "PROBE3-DONE $(date +%H:%M)" >> "$S"
