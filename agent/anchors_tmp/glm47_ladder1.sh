#!/bin/bash
# Flash T3 memory descent, round 1 (user directive: drive memory as low as
# possible). Baseline standing: T3@192k·b5 = 158.8 GiB (attention bwd
# transient 83.3, experts 24.8, slack 36.2). Probes: sdparecomp, expact, both.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
echo "GLM47-LADDER1 begin $(date +%H:%M)" >> "$S"
run_cell l47sdpa glm4.7-flash "$T3TOK" 192000 "5" "none|false|false|false|false|true"
run_cell l47exp  glm4.7-flash "$T3TOK" 192000 "5" "none|true|false|false|false|false"
run_cell l47both glm4.7-flash "$T3TOK" 192000 "5" "none|true|false|false|false|true"
echo "GLM47-LADDER1-DONE $(date +%H:%M)" >> "$S"
