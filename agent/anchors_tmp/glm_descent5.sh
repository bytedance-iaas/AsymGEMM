#!/bin/bash
# Descent round 5: Flash qchunk with FIXED flex kernel config (no autotune
# benchmarks -> no coherent-memory spill). Sampler armed again. Gated on
# GLM45-LADDER2-DONE.
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
export TORCHINDUCTOR_COMPILE_THREADS=1
for i in $(seq 1 1440); do grep -q "GLM45-LADDER2-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLM45-LADDER2-DONE" "$S" || { echo "DESCENT5-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "DESCENT5 begin $(date +%H:%M)" >> "$S"
bash /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/host_sampler.sh "$LOGD/host_sampler_q24k.log" &
SAMPLER=$!
export ASYMM_ATTN_QCHUNK_ROWS=24000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=24000
export ASYMM_ATTN_QCHUNK_MODE=ckpt ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_MODE=ckpt
run_cell l47q24k glm4.7-flash "$T3TOK" 192000 "5"
kill "$SAMPLER" 2>/dev/null || true
echo "DESCENT5-DONE ALL-DESCENT-COMPLETE-V2 $(date +%H:%M)" >> "$S"
