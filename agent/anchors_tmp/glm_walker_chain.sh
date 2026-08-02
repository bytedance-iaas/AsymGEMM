#!/bin/bash
# Walker extension (validation-walker rule): Flash 192k·b2 baseline landed at
# 28% HBM — unprobative. Walk uns-off UP (b4,b5,b6,b7) to bracket its wall,
# then run T3 at the last-fit rung and one beyond (dominance probe).
# Air: contingent cells only fire if its b2 baseline was under-band (<60%) —
# judged by me from artifacts; here we just walk both up symmetrically after
# the earlier cells. Gated on GLMFIX2-DONE (end of current queue).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
FLASH=glm4.7-flash; AIR=glm4.5-air
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
UNSOFF="superoffload_mem|unsloth-off-ohbm0"

for i in $(seq 1 1440); do grep -q "ALL-GLM-DEV-RERUNS-COMPLETE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "ALL-GLM-DEV-RERUNS-COMPLETE" "$S" || { echo "WALKER-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "GLM-WALKER begin $(date +%H:%M)" >> "$S"

# Flash uns-off up-walk: find last fit
last_fit=0
for b in 4 5 6 7; do
  v=$(run_cell gw47o$b $FLASH "$UNSOFF" 192000 "$b")
  if [ "$v" = "TRAINED" ]; then last_fit=$b; else break; fi
done
echo "WALKER flash uns-off last_fit=b$last_fit $(date +%H:%M)" >> "$S"

# T3 at baseline's last fit and one beyond (dominance probe)
if [ "$last_fit" -ge 4 ]; then
  run_cell gw47t$last_fit $FLASH "$T3TOK" 192000 "$last_fit"
  nb=$((last_fit+1))
  run_cell gw47t$nb $FLASH "$T3TOK" 192000 "$nb"
fi
echo "GLM-WALKER-DONE $(date +%H:%M)" >> "$S"
