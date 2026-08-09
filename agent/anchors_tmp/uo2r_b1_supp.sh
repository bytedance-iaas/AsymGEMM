#!/bin/bash
# Supplement: b1 probes the main uo2r chain never tried (its dense batch lists
# bottomed at b2). Waits for UO2R_DONE, then walks q3-32b (and llama3.3-70b if
# its rungs also walled above b1) at b1 per panel rung until the true wall.
set -uo pipefail
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/uo2r_status.log"
until grep -q 'UO2R_DONE' "$S2" 2>/dev/null; do sleep 60; done

export GPU="0,1" HOSTFLOOR=1300
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
export GPU_POOL="0,1"
UO="superoffload_mem|unsloth-off-ohbm0"
POL="none|false|false|false|false|false"

cell() { local v; v=$(run_cell "$1" "$2" "$UO" "$3" "1" "$POL" 2); echo "UO2R CELL $1 $2 s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"; echo "$v"; }

# q3-32b b1 ladder (128k walled at b2 in the main chain)
for s in 128000 168000 256000 320000 384000 416000; do
  v=$(cell "uo32b1s$((s/1000))" q3-32b "$s")
  [ "$v" = "TRAINED" ] || { echo "UO2R 32b b1 wall at $s ($v)" >> "$S2"; break; }
done

# llama3.3-70b b1 ladder ONLY if the main chain walled without reaching b1
if grep -q 'UO2R 70b wall at 104000' "$S2"; then
  for s in 104000 128000 168000 192000 224000 256000; do
    v=$(cell "uo70b1s$((s/1000))" llama3.3-70b "$s")
    [ "$v" = "TRAINED" ] || { echo "UO2R 70b b1 wall at $s ($v)" >> "$S2"; break; }
  done
fi
echo "UO2R_B1_DONE $(date +%H:%M:%S)" >> "$S2"
