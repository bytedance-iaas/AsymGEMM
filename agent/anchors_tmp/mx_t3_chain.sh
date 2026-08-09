#!/bin/bash
# MIXTRAL T3 (ker101) enablement chain — mirrors hy_t3_chain.
# GATE0 numeric probe at mixtral shapes; GATE1 real-model T3-vs-T2B pair @16k;
# GATE2 tput A/B @320k (house rung: T2B 594 crowned, generic-T3 was 506);
# GATE3 ceiling probes 352k/384k (448k measured all-dead host rung upstream).
# 1-rank only (mixtral house figure is 1r; 2r is the other session's lane).
set -uo pipefail
S="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/mx_t3_status.log"
export GPU=0 HOSTFLOOR=1000
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
POL="none|false|false|false|false|false"

echo "MX GATE0 numeric $(date +%H:%M:%S)" >> "$S"
.venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --mixtral --tokens 4096 \
  > agent/anchors_tmp/mx_t3_numeric.log 2>&1
rc=$?
.venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --mixtral --tokens 4096 --zero-b \
  >> agent/anchors_tmp/mx_t3_numeric.log 2>&1
rc2=$?
echo "MX GATE0 rc=${rc}/${rc2} $(tail -1 agent/anchors_tmp/mx_t3_numeric.log | head -c 100)" >> "$S"
if [ $rc -ne 0 ] || [ $rc2 -ne 0 ]; then echo "MX GATE0 FAILED — abort" >> "$S"; exit 1; fi

cell() { # $1 tag $2 systok $3 seq
  local v
  v=$(run_cell "$1" mixtral-8x22b "$2" "$3" "1" "$POL" 1)
  echo "MX CELL $1 s=$3 -> $v $(date +%H:%M:%S)" >> "$S"
  echo "$v"
}

echo "MX GATE1 pair @16k $(date +%H:%M:%S)" >> "$S"
cell mx3cor_t3 "asym_cpuadamwds|T3" 16000
cell mx3cor_t2b "asym_cpuadamwds|T2B" 16000

echo "MX GATE2 tput 320k $(date +%H:%M:%S)" >> "$S"
cell mx3tp320 "asym_cpuadamwds|T3" 320000

for s in 352000 384000; do
  v=$(cell "mx3c$((s/1000))" "asym_cpuadamwds|T3" "$s")
  [ "$v" = "TRAINED" ] || { echo "MX T3 r1 wall at $s ($v)" >> "$S"; break; }
done
echo "MX_T3_DONE $(date +%H:%M:%S)" >> "$S"
