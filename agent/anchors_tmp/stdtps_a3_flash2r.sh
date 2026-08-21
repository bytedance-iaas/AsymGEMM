#!/bin/bash
# STDTPS Agent-3 phase 3: GLM-4.7-Flash 2-rank grid fill — new 384k column
# (grid 256-896k; 256/512/640/768/896k banked). GPUs 0+1, bx2 protocol
# (HOSTFLOOR 1200, DDP_TIMEOUT 1500). Cells: rc + fsdp2 in-bracket probes
# ((320k,416k] walls), un (fits expected — 416k banked), asym sdp2-T1 seed
# b2 (416k banked b2) then b1. zero3 = derived-if-rc-fits (row convention,
# banked at edit time, no run).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1" DDP_TIMEOUT=1500
unset ASYM_ARENA_SHM_CAP_GB || true
POL="none|false|false|false|false|false"

cell() { local v; v=$(run_cell "$1" glm4.7-flash "$2" "$3" "$4" "$POL" 2)
  echo "STDTPS-A3 CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

echo "STDTPS-A3-FLASH2R BEGIN $(date '+%F %H:%M:%S')" >> "$S"
cell st3fl2rc384 "superoffload_mem|recomp"            384000 "1"   >/dev/null  # probe (320k,416k]
cell st3fl2fd384 "fsdp2_offload|recomp"               384000 "1"   >/dev/null  # probe (320k,416k]
cell st3fl2un384 "superoffload_mem|unsloth"           384000 "1"   >/dev/null
shm_guard
cell st3fl2t1384 "asym_sdp2_cpuadamwds|T1"            384000 "2 1" >/dev/null
echo "STDTPS-A3-FLASH2R-DONE $(date '+%F %H:%M:%S')" >> "$S"
