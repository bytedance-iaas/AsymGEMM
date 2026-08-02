#!/bin/bash
# Combined fixed descent chain (contiguity fix landed): Flash q-chunk cells,
# then Air q-chunk cells + Air b4 capacity probe. Fresh tags.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
echo "DESCENT2 begin $(date +%H:%M)" >> "$S"

export ASYMM_ATTN_QCHUNK_ROWS=24000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=24000
run_cell l47q24f glm4.7-flash "$T3TOK" 192000 "5"
export ASYMM_ATTN_QCHUNK_ROWS=12000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=12000
run_cell l47q12f glm4.7-flash "$T3TOK" 192000 "5"

export ASYMM_ATTN_QCHUNK_ROWS=16000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=16000
v=$(run_cell l45q16f glm4.5-air "$T3TOK" 128000 "2")
if [ "$v" = "TRAINED" ]; then
  run_cell l45q16b4 glm4.5-air "$T3TOK" 128000 "4"
fi
export ASYMM_ATTN_QCHUNK_ROWS=8000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=8000
run_cell l45q8f glm4.5-air "$T3TOK" 128000 "2"
echo "DESCENT2-DONE ALL-DESCENT-COMPLETE $(date +%H:%M)" >> "$S"
