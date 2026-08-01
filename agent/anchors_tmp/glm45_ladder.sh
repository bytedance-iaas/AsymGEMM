#!/bin/bash
# Air descent round (after Flash ladder 3): query-chunked attention against
# Air's 85-GiB attention-backward workspace (standing: 121.5 GiB @128k·b2).
# Air GQA k/v is tiny (8 heads x 128), so chunking should bite hardest here.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
for i in $(seq 1 1440); do grep -q "GLM47-LADDER3-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLM47-LADDER3-DONE" "$S" || { echo "LADDER45-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "GLM45-LADDER begin $(date +%H:%M)" >> "$S"
export ASYMM_ATTN_QCHUNK_ROWS=16000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=16000
v24=$(run_cell l45q16 glm4.5-air "$T3TOK" 128000 "2")
if [ "$v24" = "TRAINED" ]; then
  run_cell l45q16b4 glm4.5-air "$T3TOK" 128000 "4"
fi
export ASYMM_ATTN_QCHUNK_ROWS=8000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=8000
run_cell l45q8 glm4.5-air "$T3TOK" 128000 "2"
echo "GLM45-LADDER-DONE ALL-DESCENT-COMPLETE $(date +%H:%M)" >> "$S"
