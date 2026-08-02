#!/bin/bash
# Air descent round 2: attention is capped (85->20 via qchunk16k); peak is now
# expert-backward-bound (~125 total). Probe tighter moefg elementwise chunks.
# Gated on DESCENT4-DONE (end of current queue).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
export TORCHINDUCTOR_COMPILE_THREADS=1
for i in $(seq 1 1440); do grep -q "DESCENT4-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "DESCENT4-DONE" "$S" || { echo "LADDER45B-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "GLM45-LADDER2 begin $(date +%H:%M)" >> "$S"
export ASYMM_ATTN_QCHUNK_ROWS=16000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=16000
export ASYMM_FG_ELEMENTWISE_CHUNK_MB=256 ASYM_GEMM_LF_CONFIG_ASYMM_FG_ELEMENTWISE_CHUNK_MB=256
run_cell l45qc256 glm4.5-air "$T3TOK" 128000 "2"
export ASYMM_FG_ELEMENTWISE_CHUNK_MB=128 ASYM_GEMM_LF_CONFIG_ASYMM_FG_ELEMENTWISE_CHUNK_MB=128
run_cell l45qc128 glm4.5-air "$T3TOK" 128000 "2"
echo "GLM45-LADDER2-DONE $(date +%H:%M)" >> "$S"
