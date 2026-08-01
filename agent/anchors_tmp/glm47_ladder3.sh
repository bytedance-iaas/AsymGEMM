#!/bin/bash
# Ladder round 3: query-chunked flex attention (the structural lever for the
# 83-GiB MLA attention-backward transient). rows=24k then 12k at 192k·b5.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
for i in $(seq 1 1440); do grep -q "GLM47-LADDER2-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLM47-LADDER2-DONE" "$S" || { echo "LADDER3-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
export ASYMM_ATTN_QCHUNK_ROWS=24000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=24000
run_cell l47q24 glm4.7-flash "$T3TOK" 192000 "5"
export ASYMM_ATTN_QCHUNK_ROWS=12000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=12000
run_cell l47q12 glm4.7-flash "$T3TOK" 192000 "5"
echo "GLM47-LADDER3-DONE $(date +%H:%M)" >> "$S"
