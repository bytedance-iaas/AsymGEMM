#!/bin/bash
# fig12 probe 7b — naive-arm batch walks (best-feasible-batch comparison):
# naive @96k walks 6->4->2 (b8 measured GOOM in probe7).
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-500}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POLN="none|false|false|false|false|false"
echo "PROBE7B begin $(date +%H:%M)" >> "$S"
ARM_ENV="" run_cell p7n96w q3-30b-a3b "asym_cpuadamwds|unsloth" 96000 "6 4 2" "$POLN" 1
# 122B third length: 288k pair (b1; on the fig8 grid)
ARM_ENV="" run_cell p7n122_288 q3.5-122b-a10b "asym_cpuadamwds|unsloth" 288000 "1" "$POLN" 1
ARM_ENV="" run_cell p7a122_288 q3.5-122b-a10b "asym_cpuadamwds|T1" 288000 "1" "$POLN" 1
echo "PROBE7B-DONE $(date +%H:%M)" >> "$S"
