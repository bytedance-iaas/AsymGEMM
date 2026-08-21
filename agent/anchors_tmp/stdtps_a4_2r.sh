#!/bin/bash
# STDTPS Agent-4 phase 2: Hunyuan-A13B 2-rank grid fill — new rungs 160k /
# 224k (grid 160-320k; 192/256/288/320k banked). GPUs 0+1, sdp2 shared
# fabric, HY_CAMPAIGN protocol. NOTE GPU1 carries ~9.4 GB of another user's
# job (~180 GB free): any GOOM at the HBM edge is INCONCLUSIVE — flag, don't
# bank as a wall without the deficit note.
#   160k: rc+un fit expected (192k banked fits); asym sdp2-T1 seed b2 then b1.
#   224k: rc = in-bracket probe (192k,256k]; un fits expected (256k banked);
#         asym sdp2-T1 b1.
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_lib.sh
export CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
RC="superoffload_mem|recomp"
UN="superoffload_mem|unsloth-ohbm0"
ASY2="asym_sdp2_cpuadamwds|T1"

cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "$4" "$POL" 2)
  echo "STDTPS-A4 CELL $1 r2 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

echo "STDTPS-A4-2R BEGIN $(date '+%F %H:%M:%S')" >> "$S"
# ---- 160k ----
cell st4hy2rc160 "$RC"   160000 "1"    >/dev/null
cell st4hy2un160 "$UN"   160000 "1"    >/dev/null
# uo probe (c14 session's addition, adopted): 2r uo wall bracket (128k,192k]
# spans 160k -> must probe before labeling (complete-variant cell; MAIN-dropped)
cell st4hy2uo160 "superoffload_mem|unsloth-off-ohbm0" 160000 "1" >/dev/null
shm_guard
# b2 and b1 as SEPARATE cells: the 1r b2@160k point trained edge-taxed
# (98.7% resv, 1039 < the b1 curve) — max-TP needs both points when the
# higher batch sits at the edge; cell = best of the two at bank time.
cell st4hy2t1160 "$ASY2" 160000 "2"    >/dev/null
shm_guard
cell st4hy2t1160b1 "$ASY2" 160000 "1"  >/dev/null
# ---- 224k ----
cell st4hy2rc224 "$RC"   224000 "1"    >/dev/null   # probe: bracket (192k,256k]
cell st4hy2un224 "$UN"   224000 "1"    >/dev/null
shm_guard
cell st4hy2t1224 "$ASY2" 224000 "1"    >/dev/null
echo "STDTPS-A4-2R-DONE $(date '+%F %H:%M:%S')" >> "$S"
