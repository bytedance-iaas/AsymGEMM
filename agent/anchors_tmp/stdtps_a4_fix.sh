#!/bin/bash
# STDTPS Agent-4 fix chain: asym T1 @160k 1r at b1 — the b2 cell TRAINED at
# 98.7% resv = edge-taxed (1039 eff); max-TP rule needs the b1 point (expected
# ~1450, between banked T1 1559@128k and 1154@192k). Cell = best of the two.
set -uo pipefail
export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/stdtps_lib.sh
export CUDA_VISIBLE_DEVICES=0
unset GPU_POOL DDP_TIMEOUT || true
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"
echo "STDTPS-A4-FIX BEGIN $(date '+%F %H:%M:%S')" >> "$S"
v=$(run_cell st4hy1t1160b1 hunyuan-a13b "asym_cpuadamwds|T1" 160000 "1" "$POL" 1)
echo "STDTPS-A4 CELL st4hy1t1160b1 r1 asym_cpuadamwds s=160000 -> $v $(date +%H:%M:%S)" >> "$S"
echo "STDTPS-A4-FIX-DONE $(date '+%F %H:%M:%S')" >> "$S"
