#!/bin/bash
# STDTPS Agent-4 phase 1: Hunyuan-A13B 1-rank grid fill — new rungs 160k /
# 224k / 288k for the standardized 6-rung axis 160-320k (handoff:
# agent/impls/s04-p1-dgx-02-c06/standardize_tps.md). GPU0, serial, HY_CAMPAIGN
# protocol. Reused banked rungs: 192k / 256k / 320k (no reruns).
#   160k: rc+un fit expected (192k banked fits); asym T1 seed b2 (=320k tokens,
#         T1's banked last fit) then b1.
#   224k: rc = in-bracket probe (192k,256k]; un fits expected; asym T1 b1.
#   288k: un = THE handoff probe (bracket (256k,320k]); asym T1 b1 (320k fits).
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_lib.sh
export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT || true
# hunyuan ties embed/lm_head; asym offload stage rejects tied pairs -> exclude
# embeddings; router (HunYuanMoEV1Gate wrapper) kept intact on GPU (HY_CAMPAIGN).
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
RC="superoffload_mem|recomp"
UN="superoffload_mem|unsloth-ohbm0"
ASY1="asym_cpuadamwds|T1"

cell() { local v; v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "$4" "$POL" 1)
  echo "STDTPS-A4 CELL $1 r1 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S"; echo "$v"; }

echo "STDTPS-A4-1R BEGIN $(date '+%F %H:%M:%S')" >> "$S"
# ---- 160k ----
cell st4hy1rc160 "$RC"   160000 "1"    >/dev/null
cell st4hy1un160 "$UN"   160000 "1"    >/dev/null
cell st4hy1t1160 "$ASY1" 160000 "2 1"  >/dev/null
# ---- 224k ----
cell st4hy1rc224 "$RC"   224000 "1"    >/dev/null   # probe: bracket (192k,256k]
cell st4hy1un224 "$UN"   224000 "1"    >/dev/null
cell st4hy1t1224 "$ASY1" 224000 "1"    >/dev/null
# ---- 288k ----
cell st4hy1un288 "$UN"   288000 "1"    >/dev/null   # handoff probe (256k,320k]
cell st4hy1t1288 "$ASY1" 288000 "1"    >/dev/null
echo "STDTPS-A4-1R-DONE $(date '+%F %H:%M:%S')" >> "$S"
