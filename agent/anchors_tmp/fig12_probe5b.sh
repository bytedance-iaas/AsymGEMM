#!/bin/bash
# fig12 probe 5b — 30B ratio length at 800k (naive-T1's last plausible fit):
# B-T1 @800k, A-T2 @800k (walk T2B on OOM). Dataset auto-builds inline.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|false|false|false|false"
POLA="none|false|true|false|false|false"
echo "PROBE5B begin $(date +%H:%M)" >> "$S"
v=$(ARM_ENV="" run_cell nb800 q3-30b-a3b "asym_cpuadamwds|T1" 800000 "1" "$POL" 1)
echo "PROBE5B B-T1 800k -> $v" >> "$S"
if [ "$v" != "TRAINED" ]; then
  ARM_ENV="" run_cell nb700 q3-30b-a3b "asym_cpuadamwds|T1" 700000 "1" "$POL" 1
fi
v=$(ARM_ENV="" run_cell na800 q3-30b-a3b "asym_cpuadamwds|T2" 800000 "1" "$POLA" 1)
if [ "$v" != "TRAINED" ]; then
  ARM_ENV="" run_cell na800b q3-30b-a3b "asym_cpuadamwds|T2B" 800000 "1" "$POLA" 1
fi
echo "PROBE5B-DONE $(date +%H:%M)" >> "$S"
