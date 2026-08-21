#!/bin/bash
# STDTPS Agent-3 phase 2: GLM-4.7-Flash 1-rank grid fill — new asym cells
# 512k / 768k / 896k for the 256-896k grid (256/384/640k banked; un@512k=310
# and uo@896k=172 reused from the banked run_glms comment cells; baselines
# beyond their measured walls = OOM by monotonicity — NO baseline runs here).
# GPU0, serial, glmext protocol (HOSTFLOOR 500). Tier logic:
#   512k: T1 (T1@640k banked fits at 148.6 GiB).
#   768k: T1 first (T1 wall unknown in (640k,1024k]); on OOM -> T2.
#   896k: T1 only if T1@768k trained, else straight T2 (T2@1024k banked fits).
set -uo pipefail
export GPU=0 HOSTFLOOR=500
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_lib.sh
export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
POL="none|false|false|false|false|false"

cell() { local v; v=$(run_cell "$1" glm4.7-flash "$2" "$3" "$4" "$POL" 1)
  echo "STDTPS-A3 CELL $1 r1 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

echo "STDTPS-A3-FLASH1R BEGIN $(date '+%F %H:%M:%S')" >> "$S"
cell st3fl1t1512 "asym_cpuadamwds|T1" 512000 "1" >/dev/null
v768=$(cell st3fl1t1768 "asym_cpuadamwds|T1" 768000 "1")
[ "$v768" != "TRAINED" ] && cell st3fl1t2768 "asym_cpuadamwds|T2" 768000 "1" >/dev/null
if [ "$v768" = "TRAINED" ]; then
  v896=$(cell st3fl1t1896 "asym_cpuadamwds|T1" 896000 "1")
  [ "$v896" != "TRAINED" ] && cell st3fl1t2896 "asym_cpuadamwds|T2" 896000 "1" >/dev/null
else
  cell st3fl1t2896 "asym_cpuadamwds|T2" 896000 "1" >/dev/null
fi
echo "STDTPS-A3-FLASH1R-DONE $(date '+%F %H:%M:%S')" >> "$S"
