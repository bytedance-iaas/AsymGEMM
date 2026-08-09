#!/bin/bash
# Final 2-rank probes on CLEAN shm (leaked fabric segments removed):
# 1) T2B@320k arena=320 (uncontaminated verdict; shm 479G can hold ~300G) —
#    if TRAINED continue 384k; if COOM/GOOM that is the honest tier-inversion.
# 2) 288k tiebreak pair (uns expect GOOM from 98%@256k; asym sdp2-T1 expect fit).
# Guard between cells: shm must be quiet (<5G) before each launch.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"

shm_guard() {
  local used
  for i in $(seq 1 30); do
    used=$(df -BG /dev/shm | awk 'NR==2{gsub("G","",$3); print $3}')
    [ "${used:-0}" -lt 5 ] && return 0
    rm -rf /dev/shm/asym_fabric_* 2>/dev/null || true
    sleep 10
  done
  echo "HY shm_guard timeout used=${used}G" >> "$S2"
}

cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" 2); echo "HY CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"; echo "$v"; }

# 1) clean T2B attempt
export ASYM_ARENA_SHM_CAP_GB=320
shm_guard
v=$(cell hyx5_a2b320 "asym_sdp2_cpuadamwds|T2B" 320000)
if [ "$v" = "TRAINED" ]; then
  shm_guard; v=$(cell hyx5_a2b384 "asym_sdp2_cpuadamwds|T2B" 384000)
  [ "$v" = "TRAINED" ] && { shm_guard; cell hyx5_a2b448 "asym_sdp2_cpuadamwds|T2B" 448000; }
else
  echo "HY r2 T2B wall at 320000 ($v) [clean shm, arena320]" >> "$S2"
fi
unset ASYM_ARENA_SHM_CAP_GB

# 2) 288k tiebreak pair
shm_guard
cell hy288b_un "superoffload_mem|unsloth-ohbm0" 288000
shm_guard
cell hy288b_asy "asym_sdp2_cpuadamwds|T1" 288000
echo "HY_FINAL_DONE $(date +%H:%M:%S)" >> "$S2"
